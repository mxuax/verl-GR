import numpy as np
from pytest import approx

from verl_gr.recipes.openonerec.onerec_profile_metrics import (
    _select_openonerec_mean_metric,
    add_openonerec_eval_aliases,
    compute_openonerec_training_reward_metrics,
)


class _Batch:
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


def test_compute_openonerec_training_reward_metrics():
    batch = _Batch(
        {
            "hit_reward": np.array([1.0, 0.0, 0.5], dtype=np.float32),
            "score": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "partial_hit_reward": np.array([10.0, 0.0, 5.0], dtype=np.float32),
            "format_reward": np.array([1.0, 0.0, 1.0], dtype=np.float32),
            "pass_rate": np.array([1.0, 1.0, 0.0], dtype=np.float32),
        }
    )

    metrics = compute_openonerec_training_reward_metrics(batch)
    assert metrics["train/openonerec/reward_total"] == approx(0.5)
    assert metrics["train/reward_total"] == approx(0.5)
    assert metrics["train/openonerec/reward"] == approx(1.0 / 3.0)
    assert metrics["train/openonerec/hit_any"] == approx(2.0 / 3.0)
    assert metrics["train/openonerec/partial_hit_reward"] == approx(5.0)
    assert metrics["train/openonerec/pass_rate"] == approx(2.0 / 3.0)
    assert metrics["train/openonerec/format_reward"] == approx(2.0 / 3.0)


def test_compute_openonerec_training_reward_metrics_pass_at_1_fallback():
    """When ``score`` is missing, ``pass_at_1`` should be used for reward."""
    batch = _Batch(
        {
            "hit_reward": np.array([1.0, 1.0], dtype=np.float32),
            "pass_at_1": np.array([0.0, 1.0], dtype=np.float32),
        }
    )
    metrics = compute_openonerec_training_reward_metrics(batch)
    assert metrics["train/openonerec/reward"] == approx(0.5)
    assert metrics["train/reward"] == approx(0.5)


def test_select_openonerec_mean_metric_val_aux():
    metric_dict = {
        "val-aux/onerec/score/best@32/mean": 0.25,
        "val-aux/onerec/score/best@16/mean": 0.30,
    }
    assert _select_openonerec_mean_metric(metric_dict, "score") == approx(0.25)  # picks higher N


def test_select_openonerec_mean_metric_val_core():
    metric_dict = {
        "val-core/onerec/hit_reward/best@32/mean": 0.40,
    }
    assert _select_openonerec_mean_metric(metric_dict, "hit_reward") == approx(0.40)


def test_select_openonerec_mean_metric_flat_mean():
    """4-part keys: val-aux/.../var_name/mean (from pass_at_k helpers)."""
    metric_dict = {
        "val-aux/onerec/pass_at_32/mean": 0.55,
    }
    assert _select_openonerec_mean_metric(metric_dict, "pass_at_32") == approx(0.55)


def test_select_openonerec_mean_metric_mean_at():
    """Keys with 'mean@N' format."""
    metric_dict = {
        "val-aux/onerec/score/mean@8": 0.12,
        "val-aux/onerec/score/mean@16": 0.18,
    }
    assert _select_openonerec_mean_metric(metric_dict, "score") == approx(0.18)


def test_select_openonerec_mean_metric_missing_var():
    metric_dict = {"val-aux/onerec/other/best@32/mean": 0.1}
    assert _select_openonerec_mean_metric(metric_dict, "score") is None


def test_add_openonerec_eval_aliases():
    metric_dict = {
        "val-aux/onerec/score/best@32/mean": 0.25,
        "val-aux/onerec/score/mean@32": 0.11,
        "val-aux/onerec/hit_reward/best@32/mean": 0.4,
        "val-aux/onerec/hit_reward/mean@32": 0.05,
        # Stale larger-N mean must not win when preferred_n=32.
        "val-aux/onerec/hit_reward/mean@1024": 0.99,
    }
    add_openonerec_eval_aliases(
        metric_dict,
        preferred_n=32,
        n_prompts=100,
        n_responses=3200,
    )
    assert metric_dict["eval/reward"] == 0.11
    assert metric_dict["eval/reward_total"] == 0.05
    assert metric_dict["eval/n_prompts"] == 100.0
    assert metric_dict["eval/n_responses_per_prompt"] == 32.0


def test_add_openonerec_eval_aliases_val_core():
    """Aliases should be resolved from val-core/ keys as well."""
    metric_dict = {
        "val-core/onerec/score/best@32/mean": 0.30,
        "val-aux/onerec/hit_reward/best@32/mean": 0.45,
    }
    add_openonerec_eval_aliases(metric_dict)
    assert metric_dict["eval/reward"] == 0.30
    assert metric_dict["eval/reward_total"] == 0.45


def test_add_openonerec_eval_aliases_fallback():
    """reward_total should fall back to reward when hit_reward is missing."""
    metric_dict = {
        "val-aux/onerec/score/best@32/mean": 0.20,
    }
    add_openonerec_eval_aliases(metric_dict)
    assert metric_dict["eval/reward"] == 0.20
    assert metric_dict["eval/reward_total"] == 0.20  # fallback to reward
