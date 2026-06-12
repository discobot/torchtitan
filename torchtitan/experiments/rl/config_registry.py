# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config entry points for the RL/unified experiment.

Each function returns a complete ``RLTrainer.Config`` and is discoverable by
``ConfigManager`` via ``--module rl --config <function_name>``.
"""

import dataclasses
from dataclasses import dataclass

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.search_r1 import SearchR1Rollouter
from torchtitan.experiments.rl.examples.sum_digits import SumDigitsRollouter
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.trainer import GRPOLoss, RLTrainer
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.qwen3 import model_registry
from torchtitan.protocols.model import ModelConfigConverter

_BATCH_INVARIANT_DEBUG = DebugConfig(batch_invariant=True, deterministic=True)


class BatchInvariantFlexConverter(ModelConfigConverter):
    """Pin flex attention kernel options for batch-invariant mode.

    Sets fixed BLOCK_M/BLOCK_N=16 and BACKEND=TRITON on all
    FlexAttention layers.

    BACKEND=TRITON is to avoid flex_decode kernel.
    """

    # the triton BLOCK_N tile size needs to be pinned for stable numerics and
    # needs to match vLLM's for identical results, today vLLM default is 16
    # TODO: run some experiments to determine impact of small vs large tile sizes
    _BLOCK_M = 16
    _BLOCK_N = 16

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config) -> None:
        for layer_cfg in model_config.layers:
            inner = layer_cfg.attention.inner_attention
            if isinstance(inner, FlexAttention.Config):
                inner.kernel_options["BACKEND"] = "TRITON"
                inner.kernel_options["BLOCK_M"] = self._BLOCK_M
                inner.kernel_options["BLOCK_N"] = self._BLOCK_N


def rl_grpo_qwen3_0_6b_varlen() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SumDigitsRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=True),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_flex() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B with flex attention (4 GPUs: 2 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="flex"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        # TODO: add aot_eager compiling overall, today it doesn't work because
        # we are missing mechanism to scoop Flex region to plug in inductor backend support
        rollouter=SumDigitsRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=True),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_flex_batch_invariant() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B with flex attention and batch invariance
    for bitwise-identical numerics between trainer and generator (4 GPUs: 2 gen + 2 train).
    """
    config = rl_grpo_qwen3_0_6b_flex()
    config.model_spec = model_registry(
        "0.6B",
        attn_backend="flex",
        converters=[BatchInvariantFlexConverter.Config()],
    )
    block_size = config.model_spec.model.layers[0].attention.inner_attention.block_size
    config.batcher = dataclasses.replace(
        config.batcher, per_sample_pad_multiple=block_size
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=_BATCH_INVARIANT_DEBUG,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, enable_sequence_parallel=False
        ),
    )
    config.generator = dataclasses.replace(
        config.generator, debug=_BATCH_INVARIANT_DEBUG
    )
    return config


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SumDigitsRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=True),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-0.6B, flex attention
    (4 GPUs: 2 gen + 2 train). Fast smoke config for the multi-turn search pipeline.

    Requires a running local dense retrieval server and the Search-R1 NQ/HotpotQA
    parquet data; see ``examples/search_r1/README.md``.
    """
    config = rl_grpo_qwen3_0_6b_flex()
    config.rollouter = SearchR1Rollouter.Config()
    # Text-tag protocol: keep the model's <think>/<search>/<answer> tags in the
    # completion text (no chat-template thinking scaffolding).
    config.renderer = RendererConfig(name="qwen3", enable_thinking=False)
    # Longer packed sequences to fit retrieved passages across turns.
    config.batcher = dataclasses.replace(
        config.batcher,
        batch=BatchConfig(local_batch_size=1, global_batch_size=8, seq_len=4096),
    )
    config.generator = dataclasses.replace(
        config.generator,
        sampling=SamplingConfig(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,
            stop=["</search>", "</answer>"],
        ),
    )
    config.validation_freq = 10  # held-out NQ EM every 10 steps
    config.num_validation_samples = 64  # NQ test questions per eval (less noisy EM)
    return config


def rl_grpo_qwen3_1_7b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-1.7B, varlen attention.

    Reproduces slime's ``examples/search-r1`` nq_test EM curve (~0.10 -> ~0.28+):
    **pure-EM 0/1 reward** (the rollouter's default rubric — no format/retrieval
    weighting, matching slime), **standard GRPO with group-std-normalized advantages**,
    **256 samples/step** (32 prompts x group 8, no dynamic sampling), clip-higher
    (0.2/0.28), KL-to-reference (low_var_kl, 0.001), AdamW lr 1e-6 constant,
    temperature 1.0, max 4 search turns, 500 steps. 6 GPUs: 4 gen (TP=4) + 1 train
    (TP=1) + 1 for the retriever.

    Requires a running local dense retrieval server and the Search-R1 NQ/HotpotQA
    parquet data; see ``examples/search_r1/README.md``.
    """
    config = rl_grpo_qwen3_1_7b()
    config.rollouter = SearchR1Rollouter.Config()
    config.renderer = RendererConfig(name="qwen3", enable_thinking=False)
    # seq_len 4096 fits the multi-turn rollouts. global_batch_size=48 packed rows so
    # the ~256 rollouts/step (32 prompts x 8) actually train in one optimizer step
    # (rollouts pack ~5 per 4096 row at total_len ~800), matching slime's 256.
    config.batcher = dataclasses.replace(
        config.batcher,
        batch=BatchConfig(local_batch_size=1, global_batch_size=48, seq_len=4096),
    )
    # 4 generator GPUs (TP=4) + 1 trainer GPU (TP=1) + retriever = 6 GPUs.
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=4
        ),
        sampling=SamplingConfig(
            # slime: temperature 1.0 + top_p 1.0 (no nucleus truncation).
            temperature=1.0,
            top_p=1.0,
            max_tokens=512,
            stop=["</search>", "</answer>"],
        ),
    )
    # slime stabilizers: clip-higher (0.2/0.28) + KL-to-reference (low_var_kl, 0.001).
    # lr 1e-6 constant. Trainer on 1 GPU (TP=1); generator gets 4 (TP=4).
    config.trainer = dataclasses.replace(
        config.trainer,
        loss=GRPOLoss.Config(
            clip_eps=0.2,
            clip_eps_high=0.28,
            kl_coef=0.001,
            kl_loss_type="low_var_kl",
        ),
        optimizer=default_adamw(lr=1e-6),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2, decay_type="linear", min_lr_factor=1.0
        ),
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=1
        ),
    )
    # slime recipe: 32 prompts x group_size 8 = 256 rollouts/step (= slime's
    # rollout_batch_size 32, n_samples_per_prompt 8, global_batch_size 256).
    config.num_groups_per_rollout_batch = 32
    # slime uses standard GRPO: advantages normalized by group reward std (not
    # Dr.GRPO). Up-weights hard (few-of-N correct) groups -> the search-dependent
    # questions -> the policy keeps learning to search instead of drifting closed-book.
    config.advantage_std_normalization = True
    # No dynamic sampling: slime's reference run does not filter zero-std groups.
    config.dynamic_sampling = False
    # Safety cap on the collection loop (1 round of 256 usually meets the token
    # target; cap prevents runaway if rollouts get short).
    config.max_rollout_batches_per_step = 2
    config.num_steps = 500
    # Eval on NQ test (slime-style): every 5 steps, first 500 prompts (file order).
    config.validation_freq = 5
    config.num_validation_samples = 500
    # Log a sample rollout per group each step so trajectories are inspectable.
    config.log_samples = True
    return config


def rl_grpo_qwen3_8b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-8B.

    Identical slime recipe to ``rl_grpo_qwen3_1_7b_search_r1`` (pure-EM 0/1 reward,
    standard GRPO with group-std-normalized advantages, 256 samples/step = 32
    prompts x group 8 with no dynamic sampling, clip-higher 0.2/0.28,
    KL-to-reference low_var_kl 0.001, AdamW lr 1e-6 constant, temperature 1.0 /
    top_p 1.0, max 4 search turns, 500 steps). Only the model and the GPU split
    differ — everything affecting the gradient is the same so the 8B curve is
    directly comparable to the 1.7B run.

    8 GPUs: 2 gen (TP=2) + 4 train (TP=4) + 2 for the retriever's fp16 index.
    The trainer runs fp32, so 8B needs TP=4 (~2e9 params/GPU) to avoid OOM; the
    bf16 vLLM generator fits comfortably on TP=2.

    Requires a running dense retrieval server and the Search-R1 NQ/HotpotQA
    parquet data; see ``examples/search_r1/README.md``.
    """
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model_spec = model_registry("8B", attn_backend="varlen")
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-8B"
    # Trainer fp32 8B -> TP=4 (2e9 params/GPU, ~32GB) to avoid OOM.
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=4
        ),
    )
    # Generator bf16 8B -> TP=2. 2 gen + 4 train = 6 GPUs; retriever takes 2.
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=2
        ),
    )
    return config


def rl_grpo_qwen3_14b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-14B (16 GPUs: 8 gen + 8 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("14B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-14B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SumDigitsRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=True),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=8,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=8,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL training.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SumDigitsRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=True),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # bfloat16 is needed for trainer to align with generator dtype
            # TODO: replace bfloat16 enablement with FSDP2+TP2
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=batch_invariant_config,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
            debug=batch_invariant_config,
        ),
    )
