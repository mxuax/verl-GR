import pytest
import torch

from scripts.merge_fsdp_ckpt import _merge_fsdp2_state_dicts, _remap_state_dict_for_model


def test_merge_fsdp2_state_dicts_rejects_missing_rank_key():
    shards = [
        {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.5])},
        {"weight": torch.tensor([3.0, 4.0])},
    ]

    with pytest.raises(KeyError, match="bias.*missing"):
        _merge_fsdp2_state_dicts(shards, world_size=2)


def test_remap_state_dict_for_model_rejects_missing_model_weight():
    merged = {"_fsdp_wrapped_module.layer.weight": torch.ones(2)}
    model_sd = {
        "layer.weight": torch.zeros(2),
        "layer.bias": torch.zeros(2),
    }

    with pytest.raises(RuntimeError, match="missing model keys"):
        _remap_state_dict_for_model(merged, model_sd)


def test_remap_state_dict_for_model_rejects_unmapped_checkpoint_weight():
    merged = {
        "_fsdp_wrapped_module.layer.weight": torch.ones(2),
        "_fsdp_wrapped_module.extra.weight": torch.ones(2),
    }
    model_sd = {"layer.weight": torch.zeros(2)}

    with pytest.raises(RuntimeError, match="unmapped checkpoint keys"):
        _remap_state_dict_for_model(merged, model_sd)


def test_remap_state_dict_for_model_strips_fsdp_prefix_and_casts_dtype():
    merged = {"_fsdp_wrapped_module.layer.weight": torch.ones(2, dtype=torch.float32)}
    model_sd = {"layer.weight": torch.zeros(2, dtype=torch.bfloat16)}

    remapped = _remap_state_dict_for_model(merged, model_sd)

    assert set(remapped) == {"layer.weight"}
    assert remapped["layer.weight"].dtype == torch.bfloat16
