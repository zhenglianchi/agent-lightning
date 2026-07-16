# Copyright (c) Microsoft. All rights reserved.

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

import hydra
import ray
from ray.actor import ActorClass
from verl.trainer.main_ppo import create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager

from agentlightning.adapter import TraceAdapter
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset

from .dataset import AgentDataset, LoadedDataset

if TYPE_CHECKING:
    from .daemon import AgentModeDaemon
    from .trainer import AgentLightningTrainer

__all__ = [
    "main",
    "run_ppo",
    "TaskRunner",
]


@hydra.main(config_path="pkg://agentlightning/verl", config_name="config", version_base=None)
def main(config: Any):
    from .daemon import AgentModeDaemon
    from .trainer import AgentLightningTrainer

    run_ppo(
        config,
        train_dataset=None,
        val_dataset=None,
        store=None,
        llm_proxy=None,
        adapter=None,
        trainer_cls=AgentLightningTrainer,
        daemon_cls=AgentModeDaemon,
    )


def run_ppo(
    config: Any,
    train_dataset: Dataset[Any] | None,
    val_dataset: Dataset[Any] | None,
    store: LightningStore | None,
    llm_proxy: LLMProxy | None,
    adapter: TraceAdapter[Any] | None,
    trainer_cls: Type[AgentLightningTrainer],
    daemon_cls: Type[AgentModeDaemon],
) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        num_cpus = config.ray_kwargs.ray_init.num_cpus

        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            num_cpus=num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(
        runner.run.remote(  # type: ignore
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            trainer_cls=trainer_cls,
            daemon_cls=daemon_cls,
        )
    )


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(
        self,
        config: Any,
        train_dataset: Dataset[Any] | None,
        val_dataset: Dataset[Any] | None,
        store: LightningStore | None,
        llm_proxy: LLMProxy | None,
        adapter: TraceAdapter[Any] | None,
        trainer_cls: Type[AgentLightningTrainer],
        daemon_cls: Type[AgentModeDaemon],
    ):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils.tokenizer import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
        
        # 这里verl.workers.fsdp_workers 和 verl.workers.megatron_workers 两个模块都被移除
        # 直接使用ActorRolloutRefWorker通过model_engine去自动路由fsdp和megatron
        actor_rollout_cls = ActorRolloutRefWorker
        # 这里RayWorkerGroup现在统一处理 FSDP 和 Megatron，不再需要 WorkerGroup 层面区分
        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        from verl.trainer.ppo.utils import Role

        # 这里查看是否需要注册refworker，如果需要的话也是三合一，原有的RefPolicy融合到ActorRolloutRef内部
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout

        role_worker_mapping: dict[Role, ActorClass[Any]] = {
            role: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(TrainingWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            role: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 这里在verl v0.8.0中不再使用RewardModelWorker，同时reward_model配置重构到了reward.reward_model
        # 需要在trainer.py中self.rm_wg.compute_rm_score(batch)，这个 rm_wg 是从 role_worker_mapping 里创建的 WorkerGroup
        # 修改为 0.8.0 的 self.reward_loop_manager.compute_rm_score(batch)，并在 init_workers 中初始化 reward_loop_manager 而非 rm_wg
        # 0.6.1：RewardModelWorker（FSDP/Megatron 各一个）→ 训练式推理 reward
        # 0.8.0：RewardModelManager → 启动 rollout server 做 reward 推理，不再区分 FSDP/Megatron
        if config.reward.reward_model.enable:
            # 这里没有单独的RewardModelWorker
            # role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        # 这里RefPolicy融合到ActorRolloutRef内部了，所以直接使用Role.ActorRollout
        #if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        #    role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
        #    mapping[Role.RefPolicy] = global_pool_id

        # 0.8.0 中 reward 计算改由 RewardLoopManager 管理，该 manager 在 RayPPOTrainer.init_workers 中根据 config 自动创建
        reward_fn = load_reward_manager(
            config, tokenizer, **config.reward.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, **config.reward.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Use our special dataset
        if train_dataset is None:
            train_dataset = AgentDataset(
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        else:
            train_dataset = LoadedDataset(train_dataset)

        if val_dataset is None:
            val_dataset = AgentDataset(
                data_files=config.data.val_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        else:
            val_dataset = LoadedDataset(val_dataset)

        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            daemon_cls=daemon_cls,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
