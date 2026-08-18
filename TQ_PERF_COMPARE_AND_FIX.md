# TQ 方案性能对比验证与修复方案

> 分支：`master`（TQ 接入，`AgentLightningTrainer(PPOTrainer)`）
> 对比基线：`upgrade/verl-0.8.0`（无 TQ，`AgentLightningTrainer(RayPPOTrainer)`）
> 环境：单机 16 卡（昇腾 NPU），TQ `0.1.6`，verl `release/v0.8.0` + `start_script/verl-0.8.0.patch`

## 1. 背景与已确认结论

训练对比（master vs baseline）：

| 阶段 | 观测 |
| --- | --- |
| 生成 rollout | 基本相当 |
| ref + old_log_prob | master 快约 2s |
| **update_actor** | **master 慢约 15s** |
| update_weights | master 慢约 5s |

已确认/已排除的事实：

1. **trainer 基类不同**：master 继承 `verl.trainer.main_ppo_sync.PPOTrainer`（TQ/KVBatchMeta 数据流），baseline 继承 `verl.trainer.ppo.ray_trainer.RayPPOTrainer`（DataProto 数据流）。两边 actor 的 worker 类（`ActorRolloutRefWorker`）、FSDP engine、`ppo_mini_batch_size=32`、`rollout.n=4`、`ppo_epochs` 完全一致。
2. **batch 规模一致**：三元组数量两边相同（245 左右，pad 后都是 128 的倍数），已排除。
3. **TQ 读取不是瓶颈**：`update_actor` 内 worker 侧 TQ 读取仅约 0.5s，已排除。
4. **剩余最大差异 = 训练数据的字段布局**：
   - baseline：`left_right_2_no_padding` 只把 `input_ids / position_ids / loss_mask` 转成 nested，`response_mask / old_log_probs / ref_log_prob / advantages / returns / token_level_scores / rm_scores` 都是 **padded 普通张量**；
   - master：从 TQ 读出的数据**几乎全部是 nested**（TQ 以变长 NestedTensor 存储，`_compute_*` 阶段用 `response_from_nested / response_to_nested` 写回）。

## 2. 根因假设

`ppo_loss` 中（`verl/workers/utils/losses.py:85-91`）：

```python
# select fields and convert to padded tensor
fields = ["response_mask", "old_log_probs", "advantages"]
if "ref_log_prob" in data:
    fields.append("ref_log_prob")
data = data.select(*fields).to_padded_tensor()
```

- baseline：这些字段本来就是 padded，`to_padded_tensor()` 是 no-op，每次 micro-batch 零成本；
- master：每个 micro-batch 都要对多个 nested 字段真转换，加上 `index_select_tensor_dict`、`micro_batch.to(device)`、nested 求和等操作都作用在 10+ 个 nested 字段上。在昇腾 NPU 上 `torch.nested` 算子多为慢速/回退路径，4 个 micro-batch（16 样本/卡，`micro_batch_size_per_gpu=4`）的累计开销就是 ~15s。

这解释了为什么 **infer 反而快 2s**：old_log_prob/ref 是纯前向，不吃 `ppo_loss` 的 nested→padded 路径，master 的原始变长数据还省掉了 baseline 的 pad/unpad 与 `left_right_2_no_padding` 的 GPU unpad 开销。

`update_weights` 慢 5s 是另一因素：TQ 栈（controller + 8 个 storage unit actor + store server + llm_proxy + replay buffer 轮询线程 + 每步上百次临时线程/事件循环）与 FSDP `param_offload=true / optimizer_offload=true` 的 host 侧传输/汇聚抢 CPU 和内存带宽。

## 3. 对比验证方案（先跑，确认假设）

### 3.1 插桩补丁（verl 侧，两分支通用）

在 verl 0.8.0 上加计时打点（与 `start_script/verl-0.8.0.patch` 同样方式应用）：

**a) worker 侧 `train_mini_batch` 每个 micro-batch 计时**

`verl/workers/engine_workers.py:280-301`：

```python
            for batch_idx, mini_batch_td in enumerate(dataloader):
                # ... 原有 global_token_num 逻辑 ...
                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                _mb_t0 = time.time()
                actor_output = self.train_batch(mini_batch_td)
                _mb_t1 = time.time()
                print(
                    f"[BENCH-TRAIN] rank={self.engine.get_data_parallel_rank()} "
                    f"micro_batch={batch_idx} train_batch={_mb_t1 - _mb_t0:.3f}s",
                    flush=True,
                )
                output_lst.append(actor_output)
```

**b) `ppo_loss` 内 `to_padded_tensor` 计时**

`verl/workers/utils/losses.py:85-91`：

```python
    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    _t0 = time.time()
    data = data.select(*fields).to_padded_tensor()
    _t1 = time.time()
    print(f"[BENCH-LOSS] to_padded_tensor={_t1 - _t0:.3f}s", flush=True)
```

（`verl/workers/utils/losses.py` 头部需 `import time`；critic 的 `value_loss` 同样在 `:161-162` 有一处 `to_padded_tensor`，可一并打点。）

### 3.2 运行步骤

1. baseline 分支（`upgrade/verl-0.8.0`）打上插桩，跑 3~5 个 step，保存输出：
   ```bash
   git checkout upgrade/verl-0.8.0
   # 应用插桩 patch 到 verl
   python start_algorithm.py ... > /home/ma-user/install/bench_baseline.txt 2>&1
   ```
2. master 分支（`master`）打上同样的插桩，跑同样的 step 数，保存输出。
3. 对比 `[BENCH-TRAIN]` 每个 micro-batch 的 `train_batch` 耗时与 `[BENCH-LOSS]` 的 `to_padded_tensor` 耗时。

### 3.3 判定标准

- 若 master 每个 micro-batch 的 `train_batch` 比 baseline 慢 N 秒，且 `[BENCH-LOSS] to_padded_tensor` 占其中大头 → **假设证实**，直接进入第 4 节实现方案；
- 若 `train_batch` 时间相同但整体 `update_actor` 仍慢 → 时间在 worker 侧 dispatch/collect 或 driver 侧，再配合现有 bench 的 `transit_dispatch_total / execute_total / ray_get_total / collect_total` 定位；
- 顺带记录 `padding: <n>`（master `_train_step` 已有打印）与 worker 侧 `total_data_size / mini_batch_size`，确认 batch 规模两边一致。

## 4. 实现方案（修复）

### 方案 A（推荐，最小改动）：`train_mini_batch` 入口把 loss 字段一次性转 padded

在 `verl/workers/engine_workers.py:234` `TrainingWorker.train_mini_batch` 入口（`maybe_fix_3d_position_ids(data)` 之后）加一段转换，**每个 `update_actor` 调用只转换一次**，而不是每个 micro-batch 在 `ppo_loss` 里转 4 次：

```python
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs"""
        maybe_fix_3d_position_ids(data)
        # ---- TQ perf fix: 一次性把 loss 相关 nested 字段转 padded ----
        # 保留 input_ids/position_ids 为 nested（引擎 remove-padding 前向路径需要），
        # 其余按行字段转成普通 padded 张量，避免 ppo_loss/value_loss 每 micro-batch 重复转换。
        for _key in (
            "response_mask",
            "loss_mask",
            "old_log_probs",
            "ref_log_prob",
            "advantages",
            "returns",
            "token_level_rewards",
            "token_level_scores",
            "rm_scores",
            "entropy",
        ):
            _val = data.get(_key, None)
            if isinstance(_val, torch.Tensor) and _val.is_nested:
                data[_key] = torch.nested.to_padded_tensor(_val, padding=0.0)
        # ----------------------------------------------------------------
        batch_size_per_dp = data.shape[0]
        ...
```

要点：

- `input_ids / position_ids` 保持 nested，`prepare_model_inputs` 的 `use_remove_padding` 前向路径不受影响；
- `loss_mask` 转 padded 后，`forward_backward_batch` 的 `batch_num_tokens = data["loss_mask"].sum()` 结果不变（padding 补 0）；
- `ppo_loss` 的 `to_padded_tensor()` 变成 no-op（字段已是 padded），且 `index_select_tensor_dict`、`micro_batch.to(device)` 都改为作用在普通张量上，N 个 micro-batch 全部受益；
- 该函数同时被 actor（`ActorRolloutRefWorker.update_actor`）和 critic（`TrainingWorker.train_mini_batch`）共用，一行改动两边都覆盖。

实现位置选择：

- 若希望改动只在 agent-lightning 仓库内：在 `agentlightning/verl/trainer.py` 覆盖 `_update_actor` 不可行（转换发生在 worker 进程），需要以 patch 形式打到 verl；建议直接追加到 `start_script/verl-0.8.0.patch`（该 patch 已经同时应用到两个分支的 verl 上，但修复只对 master 生效，baseline 保留原逻辑用于对照）；
- 或者独立出一个 `start_script/tq_nested_to_padded.patch`，只给 master 环境应用。

### 方案 B（可选）：`update_actor` 只读训练必需字段

`tqbridge` 读数据时通过 `KVBatchMeta.fields` 限定字段（`verl/utils/transferqueue_utils.py:270` `kv_batch_meta2batch_meta` 已支持 `select_fields`），`update_actor` 只取 `input_ids / position_ids / response_mask / old_log_probs / ref_log_prob / advantages / returns / loss_mask`，减少每 worker 物化的 nested 字段数量。

### 方案 C（update_weights 慢 5s 的缓解）

- TQ controller / storage unit 限定独立 CPU：`tq.init` 的配置里给 controller/storage 加 `num_cpus` 或 placement group；
- `SimpleStorage.num_data_storage_units` 从 8 降到 1~2，减少 ZMQ 序列化与 host 线程；
- `TQ_NUM_THREADS` 调低（默认 8）；
- 对照实验：临时关闭 store 的 `_update_tq_reward_async`（`agentlightning/store/collection_based.py:1125`）与 replay buffer 轮询（`ReplayBuffer.poll_interval`），观察 `update_weights` 是否恢复 baseline 水平，确认是否为 CPU 竞争。

## 5. 预期收益

- `update_actor`：每 step 从 ~17s 回到与 baseline 相当的 ~2-5s（省掉 4 次 micro-batch × nested→padded 转换与 nested 索引/搬运开销）；
- `update_weights`：通过方案 C 缓解 host 竞争，目标 -5s；
- 保持 master 在 ref/old_log_prob 上的 -2s 收益与 TQ 零拷贝传输的优势。

## 6. 验证闭环

修复落地后，用第 3 节的同一插桩与同一配置再跑 3~5 个 step：

1. `[BENCH-LOSS] to_padded_tensor` 应降到 ~0.000s；
2. `[BENCH-TRAIN]` 每 micro-batch 应回落到与 baseline 相同量级；
3. 整体 `update_actor` 差值应 < 1~2s；
4. 确认训练指标（`actor/loss`、`actor/grad_norm`、`global_seqlen/*`）与修复前一致，防止转换引入数值偏差。
