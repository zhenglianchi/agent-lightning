## VERL 0.6.1 → 0.8.0 升级：数据流对比与报错根因

### 关键概念

| 概念 | 值 | 含义 |
|------|-----|------|
| `data.train_batch_size` | 32 | 训练时一轮读多少个 prompt |
| `rollout.n` | 4 | 每个 prompt 让模型生成 4 个不同的 response |
| `n_gpus_per_node` | 16 | 一共 16 张 GPU |
| DP (data parallel) | 16 | 16 张 GPU 全部做数据并行，每张 GPU 处理 1/16 的数据 |
| `ppo_mini_batch_size` | 32 | agent-lightning `_train_step` 中对数据做舍入的除数 |
| `ppo_micro_batch_size_per_gpu` | 4 | 每张 GPU 在每个 forward micro-batch 中处理几条样本 |
| `world_size` | 16 | actor_rollout worker group 的总 GPU 数（纯 DP 场景等于 DP 数） |
| `pad_dataproto_to_divisor(batch, N)` | — | 把 batch 末尾补 dummy 样本，使总样本数是 N 的倍数 |

### 初始数据

```
一轮训练开始
  → 读取 32 个 prompt
  → agent daemon 把 32 个 prompt × rollout.n(4) = 128 个 response 送去 agent 环境中执行
  → agent 执行过程中：部分 response 失败（超时、错误等）→ 丢弃
  → 假设最终打包成 120 个 triplet

当前数据量：120 条 triplet
```

---

## 0.6.1 完整数据流

以下用 **128 → 120（daemon 过滤后）→ 128（pad）→ 120（unpad）→ 110（is_drop_mask）→ 96（舍入）** 的完整链条。

### 步骤 1：`pad_dataproto_to_divisor` — 填充使 batch 能被 world_size 整除

**含义**：verl 内部通过 `make_nd_compute_dataproto_dispatch_fn` 把一批数据平均分给 16 张 GPU。如果总数据量不能被 16 整除，某些 GPU 会多拿 1 条，dispatch 会报错。所以在 dispatch 之前，用 dummy 样本填充到 16 的倍数。

**算法**：
```
原始 Y = 120
divisor = world_size = 16
padded = ceil(120 / 16) × 16 = 128
pad_size = 128 - 120 = 8 条
→ 在末尾追加 8 条全 0 的 dummy triplet
→ pad_size 记录下来，后续 unpad 用
```

| 变量 | 值 |
|------|-----|
| 原始 triplet 数 | 120 |
| world_size | 16 |
| pad 后总条数 | **128** |
| pad_size | 8 |

### 步骤 2：`compute_log_prob` — 16 张 GPU 同时前向推理算 old_log_prob

**含义**：用当前 actor 模型对所有样本做前向推理，算出每个 token 的 log 概率。因为 0.6.1 不做 `pad` to 64（只 pad 到 16 的倍数），这里 dispatch 后每张 GPU 拿到的是 8 条，必须是 `micro_batch_size_per_gpu=4` 的倍数才能通过后面的 micro-batch 分片。

**dispatch 到 GPU**：
```
driver 持有 128 条 DataProto
  → make_nd_compute_dataproto_dispatch_fn 平均分给 16 张 GPU
  → GPU_0: 128/16 = 8 条
    GPU_1: 8 条
    ...
    GPU_15: 8 条
```

**每张 GPU 的计算**：
```
GPU_0 拿到 8 条 DataProto item
  → 8 条数据在 GPU_0 上做前向推理
  → 0.6.1 旧引擎：DataProto 直接迭代，没有 prepare_micro_batches 整除断言
  → 8 条 ÷ micro_batch_size(4) = 2 个 micro-batch，正常完成
  → 输出 old_log_prob: 每张 GPU 产出 8 条 log_prob
```

| 阶段 | 数据量 |
|------|--------|
| driver 侧 | 128 条（120 真实 + 8 dummy） |
| 每张 GPU | 128 ÷ 16 = 8 条 |
| micro-batch 数 | 8 ÷ 4 = 2 |

### 步骤 3：`compute_ref_log_prob` / `compute_values`（同步骤 2）

同样流程：dispatch → 每张 GPU 8 条 → 前向推理 → 旧引擎无整除断言 → 通过。

### 步骤 4：`unpad_dataproto` — 移除 dummy 样本

```
128 → unpad(pad_size=8) → 移除末尾 8 条 dummy
→ 回到 120 条真实 triplet
```

| 变量 | 值 |
|------|-----|
| unpad 后总条数 | **120** |

### 步骤 5：`compute_advantage` + `is_drop_mask` — 算优势函数 + 标记过长样本

**含义**：GRPO 算法计算每个 response 的优势值。同时 `is_drop_mask` 标记哪些 triplet 因为 prompt 过长需要丢弃。

```
120 条 triplet
  → compute_advantage: 为每条算 advantages
  → is_drop_mask 标记: 假设 10 条 prompt 过长被标记为 True
  → 过滤: 保留 ~is_drop_mask 的条目
  → 保留 120 - 10 = 110 条
```

| 变量 | 值 |
|------|-----|
| compute_advantage 前 | 120 条 |
| is_drop_mask = True | 10 条 |
| 过滤后 Y' | **110 条** |

### 步骤 6：舍入到 mini_batch_size 的倍数

**含义**：为了保证后续 PPO 更新时 mini-batch 大小整齐，丢弃尾部不足 1 个 `ppo_mini_batch_size` 的零头。

**算法**：
```
n_transition = 110
mini_batch_size = ppo_mini_batch_size = 32
n_remained = 110 // 32 × 32 = 3 × 32 = 96
丢弃: 110 - 96 = 14 条
```

| 变量 | 值 |
|------|-----|
| 舍入前 | 110 条 |
| 舍入后 | **96 条** |
| 丢弃 | 14 条 |

### 步骤 7：`update_actor` — 16 张 GPU 同时做梯度更新

**含义**：用 PPO loss 对 actor 模型做梯度更新。在 0.6.1 中，worker 初始化就把 `ppo_mini_batch_size` 归一化到 per-GPU 的值。

**0.6.1 worker 初始化时的归一化**（发生在训练开始前，只执行一次）：
```
原始 ppo_mini_batch_size = 32
归一化：32 × rollout.n(4) ÷ device_mesh.size()(16) = 128 ÷ 16 = 8
→ 归一化后 ppo_mini_batch_size_per_dp = 8
```

**dispatch 到 GPU**：
```
driver 持有 96 条 DataProto
  → dispatch_lazy_compute_data_proto 分发给 16 张 GPU
  → GPU_0: 96/16 = 6 条
    GPU_1: 6 条
    ...
    GPU_15: 6 条
```

**每张 GPU 的计算**：
```
GPU_0 拿到 6 条 DataProto item
  → 调用 DataProto.make_iterator(mini_batch_size=8)
  → 内部 convert 成 TensorDict (batch_size=6)
  → tu.make_iterator 断言: 6 % 8 == 0 ?
  → 6 % 8 != 0... 但 0.6.1 实际情况不报错
```

| 阶段 | 数据量 |
|------|--------|
| driver 侧 | 96 条 |
| 每张 GPU | 96 ÷ 16 = 6 条 |
| mini_batch_size_per_gpu | 8 |
| 6 % 8 | ≠ 0 |

**为什么 0.6.1 实测不报错？**

在 0.6.1 中，`dispatch_lazy_compute_data_proto` 处理的是 `DataProto` 对象，内部有一个关键逻辑：dispatch 时如果总数据量不能被 DP 数整除，会**裁剪或重新平衡**，保证每张 GPU 拿到的条数要么是 0 要么能满足 mini_batch 的整倍数。实际效果是：96 条 → 只分给 12 张 GPU 各 8 条，剩余 4 张 GPU 拿 0 条。拿 8 条的 GPU 上 `8 % 8 = 0` ✓。

---

## 0.8.0 完整数据流

使用完全相同的 120 条初始数据。

### 步骤 1：`pad_dataproto_to_divisor` — 同 0.6.1

```
120 → pad to 128（÷16）
```

### 步骤 2：`_compute_old_log_prob` — 前向推理算 old_log_prob

**0.8.0 的变化**：数据在 driver 侧先被转成 TensorDict（`to_tensordict` → `left_right_2_no_padding`），再 dispatch。GPU 上使用 `forward_backward_batch`，其中 `prepare_micro_batches` 有固定尺寸模式的新断言。

```
driver 侧:
  128 条 DataProto
  → to_tensordict() → TensorDict batch_size=[128]
  → left_right_2_no_padding() → TensorDict batch_size=[128]（不变）
  → make_nd_compute_dataproto_dispatch_fn 分发给 16 张 GPU

每张 GPU:
  GPU_0: 128/16 = 8 条 TensorDict
  GPU_1: 8 条
  ...
  GPU_15: 8 条

  每张 GPU 上:
    forward_backward_batch(forward_only=True)
    → prepare_micro_batches(data)
      use_dynamic_bsz = False（默认）
      micro_batch_size_per_gpu = 4（来自 actor.ppo_micro_batch_size_per_gpu）
      force_group_size = 1
      断言: per_GPU_batch(8) % (force_group_size(1) × micro_batch_size_per_gpu(4)) == 0
      → 8 % 4 = 0 ✓ 通过
```

| 阶段 | 数据量 |
|------|--------|
| driver 侧 | 128 条 |
| 每张 GPU | 128 ÷ 16 = 8 条 |
| 断言 | 8 % (1 × 4) = 0 ✓ |

✅ 在这个 case 中，pad 到 128（恰好是 64 的倍数）侥幸通过了。但如果 daemon 产出 100 条 → pad to 112 → per-GPU 7 → 7 % 4 ≠ 0 → 报错。

### 步骤 3-4：`_compute_ref_log_prob` / `_compute_values` + `unpad`

```
同 0.6.1：128 → unpad(pad_size=8) → 120
```

### 步骤 5：`compute_advantage` + `is_drop_mask`

```
120 → is_drop_mask 过滤 10 条 → 110
```

### 步骤 6：舍入

```
110 // 32 × 32 = 96
丢弃 14 条
```

### 步骤 7：`_update_actor` — 梯度更新（ERROR 2 触发点）

**0.8.0 的变化**：`_update_actor` 是 `RayPPOTrainer` 的独立方法，内部计算 `mini_batch_size = ppo_mini_batch_size × rollout.n`。

```
driver 侧:
  96 条 DataProto
  → to_tensordict() → TensorDict batch_size=[96]
  → left_right_2_no_padding() → batch_size=[96]

  → _update_actor 内部:
    ppo_mini_batch_size = 32
    ppo_mini_batch_size = 32 × rollout.n(4) = 128   ← 乘以 rollout.n!
    塞入 TensorDict: mini_batch_size = 128

  → make_nd_compute_dataproto_dispatch_fn 分发给 16 张 GPU

每张 GPU:
  GPU_0: 96/16 = 6 条 TensorDict
  GPU_1: 6 条
  ...
  GPU_15: 6 条

  每张 GPU 上:
    train_mini_batch(data):
      batch_size_per_dp = data.shape[0] = 6
      mini_batch_size = tu.pop(data, "mini_batch_size") = 128
      mini_batch_size_per_gpu = 128 / get_data_parallel_size() = 128 / 16 = 8

      make_iterator(data, mini_batch_size=8)
      → 断言: data.shape[0](6) % mini_batch_size_per_gpu(8) == 0
      → 6 % 8 != 0
      → **AssertionError: 14 % 8 != 0** ← ERROR 2!
```

| 阶段 | 数据量 |
|------|--------|
| driver 侧 | 96 条 |
| 每张 GPU | 96 ÷ 16 = 6 条 |
| mini_batch_size (driver) | 32 × 4 = 128 |
| mini_batch_size_per_gpu | 128 ÷ 16 = 8 |
| 断言 | 6 % 8 ≠ 0 ✗ |

**为什么 0.8.0 报错而 0.6.1 不报？**

| 差异点 | 0.6.1 | 0.8.0 |
|--------|-------|-------|
| mini_batch_size 在哪算 | worker 初始化归一化 | trainer `_update_actor` 方法 |
| dispatch 处理的格式 | DataProto | TensorDict |
| dispatch 是否裁剪/填充 | 是（对 DataProto 做 rebalance） | 否（TensorDict 不做额外处理） |
| `prepare_micro_batches` | 不存在 | **新增**，有整除断言 |

核心：0.6.1 的 `dispatch_lazy_compute_data_proto(DataProto)` 在分发时如果发现 96 不能被 16 张 GPU 均匀分配，会主动调整——只把数据分给 12 张卡各 8 条（8%8=0），剩下 4 张卡空转。0.8.0 的同一个 dispatch 函数处理 TensorDict 时不具备这个能力，直接把 6 条塞给每张 GPU，然后 `make_iterator` 断言失败。

---

## 两个 ERROR 的触发条件

### ERROR 1：`prepare_micro_batches`

```
文件: verl/workers/engine/utils.py:91
断言: per_GPU_batch % (force_group_size × micro_batch_size_per_gpu) == 0
即:    per_GPU_batch % (1 × 4) == 0
即:    per_GPU_batch 必须是 4 的倍数
即:    pad 后总 batch 必须是 16 × 4 = 64 的倍数
```

| daemon 产出 Y | pad 后 | per-GPU | 断言 |
|--------------|--------|---------|------|
| 14 | 16 | 1 | 1%4≠0 ✗ |
| 50 | 64 | 4 | 4%4=0 ✓ |
| 100 | 112 | 7 | 7%4≠0 ✗ |
| 120 | 128 | 8 | 8%4=0 ✓ |

### ERROR 2：`make_iterator`

```
文件: verl/utils/tensordict_utils.py:588
断言: data.shape[0] % mini_batch_size_per_gpu == 0
即:    舍入后 batch ÷ 16 % (32 × 4 ÷ 16) == 0
即:    舍入后 batch ÷ 16 % 8 == 0
即:    舍入后 batch 必须是 16 × 8 = 128 的倍数
```

| 舍入后 batch | per-GPU | 断言 |
|------------|---------|------|
| 32 | 2 | 2%8≠0 ✗ |
| 64 | 4 | 4%8≠0 ✗ |
| 96 | 6 | 6%8≠0 ✗ |
| 128 | 8 | 8%8=0 ✓ |
| 256 | 16 | 16%8=0 ✓ |

**结论**：agent daemon 产出的 batch 在 0~128 之间波动，舍入后如果能到 128 就通过，否则必报错。而 `128 = 32 × rollout.n(4)`——agent-lightning 的舍入只对齐到 32，不对齐到 128。

---

## 修改思路（纯静态 batch）

### ERROR 1：`pad_dataproto_to_divisor` 对齐到 64 而非 16

**原理**：`prepare_micro_batches` 要求 per-GPU 能被 `micro_batch_size_per_gpu(4)` 整除，即总 batch 能被 `world_size(16) × micro_batch_size_per_gpu(4) = 64` 整除。把 pad 除数从 16 改成 64。

**代码**（`_train_step` 中）：
```python
# 原来
batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

# 改为
micro_bsz = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu  # 4
divisor = self.actor_rollout_wg.world_size * micro_bsz  # 16 × 4 = 64
batch, pad_size = pad_dataproto_to_divisor(batch, divisor)
```

| daemon Y | 旧 pad | per-GPU 旧 | 新 pad | per-GPU 新 | 断言 |
|----------|--------|-----------|--------|-----------|------|
| 14 | 16 | 1 ✗ | **64** | 4 | 4%4=0 ✓ |
| 100 | 112 | 7 ✗ | **128** | 8 | 8%4=0 ✓ |
| 120 | 128 | 8 ✓ | 128 | 8 | 8%4=0 ✓ |

代价：14 → 64 比之前 (14→16) 多 pad 了 48 条 dummy。但只在 forward-only 推理阶段（没有反向传播），开销可控。

### ERROR 2：override `_update_actor`，不乘 `rollout.n`

**原理**：agent-lightning 的 batch 来自 agent daemon，已经把 `rollout.n` 条 response 展开为独立 triplet，不需要 verl 再乘一次。去掉 `× rollout.n` 后 `mini_batch_size_per_gpu = 32 ÷ 16 = 2`，比原来的 8 宽容得多。

**代码**（在 `AgentLightningTrainer` 中新增）：
```python
def _update_actor(self, batch: DataProto) -> DataProto:
    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)

    calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
        self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
    )
    distillation_use_topk = ...

    # 关键：不乘 rollout.n
    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size  # 32

    ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
    seed = self.config.actor_rollout_ref.actor.data_loader_seed
    shuffle = self.config.actor_rollout_ref.actor.shuffle

    tu.assign_non_tensor(
        batch_td,
        calculate_entropy=calculate_entropy,
        distillation_use_topk=distillation_use_topk,
        global_batch_size=ppo_mini_batch_size,
        mini_batch_size=ppo_mini_batch_size,
        epochs=ppo_epochs,
        seed=seed,
        dataloader_kwargs={"shuffle": shuffle},
        compute_loss=True,
    )
    actor_output = self.actor_rollout_wg.update_actor(batch_td)
    actor_output = tu.get(actor_output, "metrics")
    actor_output = rename_dict(actor_output, "actor/")
    actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
    actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
    return actor_output
```

**效果**：

| 舍入后 batch | per-GPU | mini_batch_size_per_gpu (新) | 断言 |
|------------|---------|---------------------------|------|
| 96 | 6 | 32÷16 = 2 | 6%2=0 ✓ |
| 64 | 4 | 2 | 4%2=0 ✓ |
| 32 | 2 | 2 | 2%2=0 ✓ |

agent-lightning 的舍入逻辑已经保证总 batch 是 32 的倍数 → per-GPU 一定是 2 的倍数 → 2%2=0 永远成立。

同法 override `_update_critic`（如果 use_critic）。

### 修改汇总

| 文件 | 修改 | 行数 |
|------|------|------|
| `agentlightning/verl/trainer.py` | `pad_dataproto_to_divisor` 除数改为 64 | 1 行 |
| `agentlightning/verl/trainer.py` | 新增 `_update_actor` override | ~30 行 |
| `agentlightning/verl/trainer.py` | 新增 `_update_critic` override（如有） | ~30 行 |
