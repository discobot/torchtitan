# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for DAPO dynamic sampling: ``RLTrainer._group_reward_std`` and the
zero-std group filter used by the controller's oversampling loop.

A group whose siblings all get the same reward has zero reward std -> all
mean-baseline advantages are 0 -> no gradient. ``dynamic_sampling`` drops such
groups (and untrainable ones) and oversamples more batches to refill the train
batch with groups that carry a learning signal.
"""

from torchtitan.experiments.rl.rollout import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.trainer import RLTrainer


def _rollout(reward, *, has_tokens=True):
    turn = RolloutTurn(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5] if has_tokens else [],
        completion_logprobs=[-0.1, -0.2] if has_tokens else [],
        completion_message={"role": "assistant", "content": "x"},
    )
    return Rollout(
        group_id="g",
        sample_id="s",
        status=RolloutStatus.COMPLETED,
        turns=[turn],
        reward=reward,
    )


def _group(rollouts):
    return RolloutGroup(group_id="g", rollouts=rollouts)


def test_group_reward_std_mixed_is_positive():
    # Siblings with different rewards -> positive spread -> kept.
    std = RLTrainer._group_reward_std(_group([_rollout(1.0), _rollout(0.0)]))
    assert std is not None and std > 0.0


def test_group_reward_std_uniform_is_zero():
    # All siblings same reward -> zero std -> no learning signal.
    std = RLTrainer._group_reward_std(_group([_rollout(1.0), _rollout(1.0)]))
    assert std == 0.0


def test_group_reward_std_untrainable_is_none():
    # A sibling with no assistant tokens makes the group untrainable.
    std = RLTrainer._group_reward_std(
        _group([_rollout(1.0), _rollout(0.0, has_tokens=False)])
    )
    assert std is None


def test_group_reward_std_missing_reward_is_none():
    # A sibling that was never scored (reward None) makes the group untrainable.
    std = RLTrainer._group_reward_std(_group([_rollout(1.0), _rollout(None)]))
    assert std is None


def test_dynamic_sampling_filter_keeps_only_signal_groups():
    # The controller keeps groups with (std or 0.0) > 0.0; this drops both
    # zero-std (uniform reward) and untrainable (std None) groups.
    groups = [
        _group([_rollout(1.0), _rollout(0.0)]),  # signal -> kept
        _group([_rollout(1.0), _rollout(1.0)]),  # zero std -> dropped
        _group([_rollout(0.0), _rollout(0.0)]),  # zero std -> dropped
        _group([_rollout(1.0), _rollout(0.0, has_tokens=False)]),  # untrainable
    ]
    kept = [g for g in groups if (RLTrainer._group_reward_std(g) or 0.0) > 0.0]
    assert len(kept) == 1
    assert RLTrainer._group_reward_std(kept[0]) > 0.0
