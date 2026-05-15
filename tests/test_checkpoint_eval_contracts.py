from __future__ import annotations

import ast
from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _get_function(module: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"function {function_name} not found")


def test_merge_fsdp_checkpoint_fails_fast_on_partial_model_match():
    source = _read_text("scripts/merge_fsdp_ckpt.py")
    module = ast.parse(source)
    helper = _get_function(module, "_remap_state_dict_for_model")
    merge_fn = _get_function(module, "merge")
    helper_segment = ast.get_source_segment(source, helper) or ""
    merge_segment = ast.get_source_segment(source, merge_fn) or ""

    assert "missing model keys" in helper_segment
    assert "unmapped checkpoint keys" in helper_segment
    assert "FSDP checkpoint does not exactly match" in helper_segment
    assert "_remap_state_dict_for_model(merged_sd, model_sd)" in merge_segment
    assert "strict=True" in merge_segment
    assert "strict=False" not in merge_segment


def test_merge_fsdp_checkpoint_rejects_incomplete_rank_shards():
    source = _read_text("scripts/merge_fsdp_ckpt.py")
    module = ast.parse(source)
    merge_shards_fn = _get_function(module, "_merge_fsdp2_state_dicts")
    segment = ast.get_source_segment(source, merge_shards_fn) or ""

    assert "missing_ranks" in segment
    assert "Checkpoint key" in segment
    assert "missing from rank shards" in segment
    assert "Unsupported checkpoint value" in segment


def test_eval_compare_uses_same_prefix_index_for_trie_and_processor():
    source = _read_text("scripts/eval_compare_ckpts.py")
    module = ast.parse(source)
    generate_fn = _get_function(module, "constrained_beam_generate")
    build_hash_fn = _get_function(module, "_build_hash_dict")
    generate_segment = ast.get_source_segment(source, generate_fn) or ""
    build_hash_segment = ast.get_source_segment(source, build_hash_fn) or ""

    assert "prefix_index = _constraint_prefix_index(tokenizer)" in generate_segment
    assert "_build_hash_dict(info_file, tokenizer, prefix_index=prefix_index)" in generate_segment
    assert "prefix_index = 3  # Qwen2" not in generate_segment
    assert "def _constraint_prefix_index(tokenizer)" in source
    assert "if prefix_index is None:" in build_hash_segment
    assert "prefix_index = _constraint_prefix_index(tokenizer)" in build_hash_segment
