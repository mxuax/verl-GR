from __future__ import annotations

from verl_gr.recipes.minionerec.minionerec_reward import (
    RewardPenaltyConfig,
    compute_group_training_rewards,
)


def test_empty_and_invalid_penalties_applied():
    cfg = RewardPenaltyConfig()
    completions = ["", "not_a_sid", "<a_1><b_2><c_3>\n"]
    targets = ["<a_9><b_9><c_9>\n"] * 3
    keys = ["g0", "g0", "g0"]
    out = compute_group_training_rewards(completions, targets, keys, penalty_cfg=cfg)
    assert out["shape_tags"] == ["empty", "invalid", "valid"]
    assert out["total_rewards"][0] == cfg.empty_completion
    assert out["total_rewards"][1] == cfg.invalid_sid
    assert out["total_rewards"][2] == 0.0


def test_hit_group_invalid_worse_than_valid_miss():
    cfg = RewardPenaltyConfig()
    good = "<a_1><b_1><c_1>\n"
    bad_valid = "<a_2><b_2><c_2>\n"
    bad_invalid = "garbage"
    completions = [good, bad_valid, bad_invalid]
    targets = [good, good, good]
    keys = ["g"] * 3
    out = compute_group_training_rewards(completions, targets, keys, penalty_cfg=cfg)
    assert out["rule_rewards"][0] == 1.0
    assert out["total_rewards"][2] < out["total_rewards"][1]
