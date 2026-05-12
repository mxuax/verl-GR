import ast
from pathlib import Path


def _get_function(module: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"function {function_name} not found")


def test_merge_fsdp_requires_complete_rank_shards():
    source = Path("scripts/merge_fsdp_ckpt.py").read_text()
    module = ast.parse(source)
    merge_fn = _get_function(module, "_merge_fsdp2_state_dicts")
    segment = ast.get_source_segment(source, merge_fn) or ""

    assert "missing_ranks" in segment
    assert "Refusing to merge an incomplete FSDP checkpoint" in segment
    assert "raise KeyError" in segment


def test_merge_fsdp_loads_hf_model_strictly():
    source = Path("scripts/merge_fsdp_ckpt.py").read_text()
    module = ast.parse(source)
    remap_fn = _get_function(module, "_remap_merged_state_dict")
    merge_fn = _get_function(module, "merge")
    remap_segment = ast.get_source_segment(source, remap_fn) or ""
    merge_segment = ast.get_source_segment(source, merge_fn) or ""

    assert "unexpected_keys" in remap_segment
    assert "shape_mismatches" in remap_segment
    assert "missing_keys" in remap_segment
    assert "Merged checkpoint does not exactly match the target model" in remap_segment
    assert "strict=True" in merge_segment
    assert "strict=False" not in merge_segment
