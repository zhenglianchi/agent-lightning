# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from __future__ import annotations

import math

import random
from contextlib import contextmanager
from copy import deepcopy
from pprint import pprint
from typing import Dict, Tuple, Type

import numpy as np
import torch
import verl
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
import transfer_queue as tq
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.main_ppo_sync import PPOTrainer
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from agentlightning.adapter import TraceAdapter, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore

from .daemon import AgentModeDaemon

__all__ = [
    "AgentLightningTrainer",
]


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


# This function is adapted from verl.
# We introduce a new parameter `suffix` to distinguish between metrics computed
# before and after AgentLightning鈥檚 post-processing.
# - "Before" refers to raw reward and advantage values.
# - "After" refers to values computed following post-processing, which involves:
#     (1) Dropping prompts that exceed the maximum allowed length.
#     (2) Adjusting the batch size to be a multiple of the mini PPO size.
# Different suffixes are used to label these two stages accordingly.
def compute_data_metrics(batch: DataProto, use_critic: bool = True, suffix: str = "") -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean" + suffix: torch.mean(sequence_score).detach().item(),
        "critic/score/max" + suffix: torch.max(sequence_score).detach().item(),
        "critic/score/min" + suffix: torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean" + suffix: torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max" + suffix: torch.max(sequence_reward).detach().item(),
        "critic/rewards/min" + suffix: torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean" + suffix: torch.mean(valid_adv).detach().item(),
        "critic/advantages/max" + suffix: torch.max(valid_adv).detach().item(),
        "critic/advantages/min" + suffix: torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean" + suffix: torch.mean(valid_returns).detach().item(),
        "critic/returns/max" + suffix: torch.max(valid_returns).detach().item(),
        "critic/returns/min" + suffix: torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean" + suffix: torch.mean(valid_values).detach().item(),
                "critic/values/max" + suffix: torch.max(valid_values).detach().item(),
                "critic/values/min" + suffix: torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var" + suffix: (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean" + suffix: torch.mean(response_length).detach().item(),
        "response_length/max" + suffix: torch.max(response_length).detach().item(),
        "response_length/min" + suffix: torch.min(response_length).detach().item(),
        "response_length/clip_ratio"
        + suffix: torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean" + suffix: torch.mean(prompt_length).detach().item(),
        "prompt_length/max" + suffix: torch.max(prompt_length).detach().item(),
        "prompt_length/min" + suffix: torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio"
        + suffix: torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


class AgentLightningTrainer(PPOTrainer):
    """
    Specialized PPO trainer for agent-based reinforcement learning.

    This trainer is designed specifically for scenarios where the model interacts with
    external environments, tools, or APIs through an AgentLightningServer. It simplifies
    the training loop by removing the complex conditional logic present in the original
    PPOTrainer and focusing on the agent mode workflow.
    """

    def __init__(
        self,
        store: LightningStore | None,
        llm_proxy: LLMProxy | None,
        adapter: TraceAdapter | None,
        daemon_cls: Type[AgentModeDaemon],
        train_dataset=None,
        val_dataset=None,
        **kwargs,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        super().__init__(**kwargs)
        self.store = store
        self.llm_proxy = llm_proxy
        self.adapter = adapter
        self.daemon_cls = daemon_cls

    def _init_dataloader(self):
        from verl.trainer.main_ppo_sync import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn
        from torchdata.stateful_dataloader import StatefulDataLoader
        from .dataset import AgentDataset, LoadedDataset

        if self.train_dataset is not None:
            self.train_dataset = LoadedDataset(self.train_dataset)
        else:
            self.train_dataset = AgentDataset(
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config.data,
            )

        if self.val_dataset is not None:
            self.val_dataset = LoadedDataset(self.val_dataset)
        else:
            self.val_dataset = AgentDataset(
                data_files=self.config.data.val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config.data,
            )

        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."

        test_data = next(iter(self.val_dataloader))
        test_batch = DataProto.from_single_dict(test_data)

        self.checkpoint_manager.wake_up_replicas()
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.llm_server_manager.get_addresses(),
            is_train=False,
        )
        self.agent_mode_daemon.run_until_all_finished()
        test_metrics = self.agent_mode_daemon.get_test_metrics()
        self.agent_mode_daemon.clear_data_and_server()
        self.checkpoint_manager.sleep_replicas()
        return test_metrics

    def _train_step(self, batch_dict: dict, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        # Isolate in a separate method to automatically recycle the variables before validation.
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        with _timer("step", timing_raw):

            # ===== 1. Rollout（daemon替代async_rollout_manager） =====
            gen_batch = batch

            # generate a batch
            with _timer("gen", timing_raw):
                self.checkpoint_manager.wake_up_replicas()
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.llm_server_manager.get_addresses()
                )
                self.agent_mode_daemon.run_until_all_finished()
                # 这里修改以后返回的已经是KVBatchMeta了
                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    partition_id="train",
                    max_prompt_length=(
                        self.config.agentlightning.trace_aggregator.trajectory_max_prompt_length
                        if self.config.agentlightning.trace_aggregator.level.startswith("trajectory")
                        else self.config.data.max_prompt_length
                    ),
                    max_response_length=(
                        self.config.agentlightning.trace_aggregator.trajectory_max_response_length
                        if self.config.agentlightning.trace_aggregator.level.startswith("trajectory")
                        else self.config.data.max_response_length
                    ),
                    device=gen_batch.batch["fake_ids"].device,
                    global_steps=self.global_steps,
                )
                metrics.update(agent_metrics)
                self.agent_mode_daemon.clear_data_and_server()
                self.checkpoint_manager.sleep_replicas()

            '''
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with _timer("gen_max", timing_raw):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]

            
            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            '''

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.reward_loop_manager.reward_loop_worker_handles is None:
                    # 这里是todo，目前verl0.8.0中也没做
                    batch = self._compute_reward_colocate(batch)

            '''if self.config.algorithm.adv_estimator == core_algos.AdvantageEstimator.REMAX:
                batch = self._add_remax_reward_baselines(batch)'''

            # ===== 2. 过滤is_drop样本（替代原来的is_drop_mask过滤） =====
            non_drop_mask = [not tag.get("is_drop", False) for tag in batch.tags]
            if not all(non_drop_mask):
                valid_indices = [i for i, m in enumerate(non_drop_mask) if m]
                metrics["training/n_triplets_prompt_too_long"] = len(batch.keys) - len(valid_indices)
                print("valid_indices: ", len(valid_indices))
                batch = KVBatchMeta(
                    keys=[batch.keys[i] for i in valid_indices],
                    tags=[batch.tags[i] for i in valid_indices],
                    partition_id=batch.partition_id,
                    fields=batch.fields,
                    extra_info=batch.extra_info,
                )

            # ===== 3. Balance batch（替代pad/unpad/floor_pad） =====
            # _balance_batch 内部会 upsample + 负载均衡
            # upsample 复制已有样本作为padding，tag标记 is_padding=True
            # 不再需要手动 pad → compute → unpad → floor_pad
            batch = self._balance_batch(batch, metrics=metrics)

            print("padding: ", len(batch.tags))

            # ===== 4. Compute old log prob =====
            # PPOTrainer版：内部计算entropy，直接 metrics.update({"actor/entropy": ...})
            # 不再需要手动从返回值提取 entropy
            with _timer("old_log_prob", timing_raw):
                batch = self._compute_old_log_prob(batch, metrics=metrics)

            # ===== 5. Compute ref log prob =====
            if self.use_reference_policy:
                with _timer("ref", timing_raw):
                    batch = self._compute_ref_log_prob(batch, metrics=metrics)

            # ===== 6. Compute values =====
            if self.use_critic:
                with _timer("values", timing_raw):
                    batch = self._compute_values(batch, metrics=metrics)

            # ===== 7. Compute advantage =====
            # PPOTrainer版：从TQ取数据 → 计算advantage → 写回TQ
            # 不再需要手动 token_level_rewards = token_level_scores
            with _timer("adv", timing_raw):
                batch = self._compute_advantage(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    batch = self._update_critic(batch, metrics=metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch = self._update_actor(batch, metrics=metrics)

        return batch

    def _cleanup(self):
        if hasattr(self, 'replay_buffer'):
            self.replay_buffer.close()
        self._shutdown_dump_executor()

    def fit(self):
        if self._dump_executor._shutdown:
            self._init_dump_executor()
            
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        if self.adapter is not None and not isinstance(self.adapter, TraceToTripletBase):
            raise ValueError("Adapter must be a TraceToTripletBase for currently VERL implementation.")
        
        model = self.config.actor_rollout_ref.model.path
        
        self.agent_mode_daemon = self.daemon_cls(
            self.config.agentlightning.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": model,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            mode="v1" if self.store is not None else "v0",
            store=self.store,
            llm_proxy=self.llm_proxy,
            adapter=self.adapter,
            processor=self.processor,  # For Qwen2-VL mrope position_ids
            image_base_dir=getattr(self.config.data, "image_base_dir", None),
            trace_aggregator=self.config.agentlightning.trace_aggregator,
        )
        self.agent_mode_daemon.start()

        try:
            if self.config.trainer.get("val_before_train", True):
                val_metrics = self._validate()
                assert val_metrics, f"{val_metrics=}"
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    return

            # add tqdm
            progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

            # we start from step 1
            self.global_steps += 1
            last_val_metrics = None

            for epoch in range(self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    metrics, timing_raw = {}, {}

                    is_last_step = self.global_steps >= self.total_training_steps

                    # train step
                    batch = self._train_step(batch_dict, metrics, timing_raw)

                    # save checkpoint
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                    # update weights from trainer to rollout
                    with _timer("update_weights", timing_raw):
                        self.checkpoint_manager.update_weights()

                    # validate
                    if self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # record metrics
                    self._compute_metrics(batch, metrics, timing_raw, global_steps=self.global_steps, epoch=epoch)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            self._log_rollout_data(batch, timing_raw, rollout_data_dir)
                
                    # cleanup transfer queue and replay buffer
                    tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1
                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        pprint(f"Flush the logger...")
                        del logger
                        pprint(f"Training finished at step {self.global_steps}.")
                        return
        finally:
            self._cleanup()
