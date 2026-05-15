from __future__ import annotations

import ast
from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _get_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found")


def _get_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return node
    raise AssertionError(f"method {method_name} not found in {class_node.name}")


def test_rl_trainer_validate_routes_to_task_adapter():
    source = _read_text("verl_gr/trainers/rl_trainer.py")
    module = ast.parse(source)
    trainer_cls = _get_class(module, "RLTrainer")
    validate_method = _get_method(trainer_cls, "_validate")

    call_found = False
    for node in ast.walk(validate_method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "validate"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "_get_task_adapter"
        ):
            call_found = True
            break
    assert call_found, "_validate should delegate to task adapter validate()"


def test_rl_trainer_reward_colocate_routes_postprocess_rewards():
    source = _read_text("verl_gr/trainers/rl_trainer.py")
    module = ast.parse(source)
    trainer_cls = _get_class(module, "RLTrainer")
    method = _get_method(trainer_cls, "_compute_reward_colocate")

    postprocess_call = False
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "postprocess_rewards"
        ):
            postprocess_call = True
            break
    assert postprocess_call, "_compute_reward_colocate should call adapter postprocess_rewards()"


def test_minionerec_validation_groups_by_uid_not_prompt_text():
    source = _read_text("verl_gr/recipes/minionerec/minionerec_trainer.py")
    module = ast.parse(source)
    adapter_cls = _get_class(module, "MiniOneRecTrainerAdapter")
    method = _get_method(adapter_cls, "_compute_ranking_metrics")

    source_segment = ast.get_source_segment(source, method) or ""
    assert "sample_uids" in source_segment
    assert "grouped_indices[(str(data_source), str(uid))]" in source_segment
    assert "sample_inputs" not in source_segment


def test_minionerec_dataset_disables_alignment_for_val_by_default():
    source = _read_text("verl_gr/recipes/minionerec/minionerec_dataset.py")
    module = ast.parse(source)
    dataset_cls = _get_class(module, "MiniOneRecDataset")
    init_method = _get_method(dataset_cls, "__init__")
    init_segment = ast.get_source_segment(source, init_method) or ""

    assert 'config.get("include_alignment_tasks_for_val", False)' in init_segment
    assert "self.is_val_split = self._is_val_split" in init_segment
    assert "self.include_alignment_tasks = (" in init_segment


def test_minionerec_agent_loop_supports_train_val_decode_modes():
    source = _read_text("verl_gr/recipes/minionerec/constrained_beam_agent_loop.py")
    module = ast.parse(source)
    worker_cls = _get_class(module, "MiniOneRecConstrainedBeamAgentLoopWorker")
    generate_fn = _get_method(worker_cls, "generate_sequences")
    segment = ast.get_source_segment(source, generate_fn) or ""

    assert 'rollout_custom.get("decode_mode_train", "stochastic_constrained")' in segment
    assert 'rollout_custom.get("decode_mode_val", "deterministic_beam")' in segment
    assert 'sampling_params[BEAM_SEARCH_PARAMS_KEY]["decode_mode"] = decode_mode' in segment
    assert 'sampling_params[BEAM_SEARCH_PARAMS_KEY]["disable_cache_in_train"]' in segment


def test_constrained_beam_server_has_stochastic_mode_branch():
    source = _read_text("verl_gr/workers/rollout/constrained_beam_vllm_async.py")
    assert 'beam_config.decode_mode == "stochastic_constrained"' in source
    assert "_run_constrained_stochastic_sample" in source


# ---------------------------------------------------------------------------
# HF branch contracts
# ---------------------------------------------------------------------------


def test_agent_loop_recognizes_hf_decode_modes():
    source = _read_text("verl_gr/recipes/minionerec/constrained_beam_agent_loop.py")
    assert '"hf_constrained_beam_sample"' in source
    assert '"hf_constrained_beam_eval"' in source


def test_worker_has_hf_beam_generate_method():
    source = _read_text("verl_gr/recipes/minionerec/minionerec_fsdp_workers.py")
    module = ast.parse(source)
    worker_cls = _get_class(module, "MiniOneRecActorRolloutRefWorker")
    method = _get_method(worker_cls, "hf_constrained_beam_generate")
    segment = ast.get_source_segment(source, method) or ""
    assert "HfConstrainedBeamGenerator" in segment
    assert "summon_full_params" in segment
    assert "my_prompt_indices" in segment  # rank sharding
    assert '"prompt_indices"' in segment


def test_hf_constrained_generator_exports_train_eval():
    source = _read_text("verl_gr/recipes/minionerec/hf_constrained_generation.py")
    assert "class HfConstrainedBeamGenerator" in source
    assert "def generate_train(" in source
    assert "def generate_eval(" in source
    assert "do_sample=True" in source or 'do_sample=True' in source
    assert "do_sample=False" in source or 'do_sample=False' in source


def test_agent_loop_manager_has_hf_generate_routing():
    source = _read_text("verl_gr/recipes/minionerec/constrained_beam_agent_loop.py")
    module = ast.parse(source)
    manager_cls = _get_class(module, "MiniOneRecConstrainedBeamAgentLoopManager")
    _get_method(manager_cls, "_resolve_hf_decode_mode")
    _get_method(manager_cls, "_should_route_to_hf")
    _get_method(manager_cls, "_hf_generate_sequences")
    assert 'prompts.meta_info.get("validate", False)' in source
    assert '"decode_mode_val"' in source
    assert '"decode_mode_train"' in source


def test_agent_loop_manager_builds_full_tensor_batch_for_hf_outputs():
    source = _read_text("verl_gr/recipes/minionerec/constrained_beam_agent_loop.py")
    manager_start = source.index("class MiniOneRecConstrainedBeamAgentLoopManager")
    segment = source[manager_start:]
    for key in ('"prompts"', '"responses"', '"input_ids"', '"attention_mask"', '"position_ids"'):
        assert key in segment
    assert "ordered_response_groups" in segment
    assert "prompt_indices" in segment


def test_minionerec_validation_uses_val_beam_width():
    source = _read_text("verl_gr/recipes/minionerec/minionerec_trainer.py")
    assert 'rollout_custom.get("val_beam_width", beam_width)' in source
    assert "repeat_times = max(1, base_generations_per_prompt) * max(1, val_beam_width)" in source


def test_yaml_uses_hf_decode_modes():
    source = _read_text("configs/verl_gr/minionerec/grpo_trainer.yaml")
    assert "hf_constrained_beam_sample" in source
    assert "hf_constrained_beam_eval" in source


def test_ddp_yaml_uses_component_defaults_not_generated_schema():
    source = _read_text("configs/verl_gr/minionerec/grpo_trainer_ddp.yaml")
    assert "/_generated_ppo_trainer" not in source
    assert "actor@actor_rollout_ref.actor: actor" in source
    assert "ref@actor_rollout_ref.ref: ref" in source
    assert "_target_: verl_gr.workers.config.ddp_engine.DDPActorConfig" in source
    assert "_target_: verl_gr.workers.config.ddp_engine.DDPEngineConfig" in source


def test_main_ppo_no_longer_rewrites_ddp_config_at_runtime():
    source = _read_text("verl_gr/trainers/main_ppo.py")
    assert "_generated_ppo_trainer" not in source
    assert "_ensure_config_defaults" not in source
    assert "_normalize_strategy_targets" not in source


def test_run_script_explicitly_forces_ddp_backend_fields():
    source = _read_text("scripts/run_minionerec_grpo.sh")
    assert '++actor_rollout_ref.actor.strategy=ddp' in source
    assert '++actor_rollout_ref.actor._target_=verl_gr.workers.config.ddp_engine.DDPActorConfig' in source
    assert '++actor_rollout_ref.actor.engine_config._target_=verl_gr.workers.config.ddp_engine.DDPEngineConfig' in source
    assert '++actor_rollout_ref.ref.strategy=ddp' in source
    assert '++actor_rollout_ref.ref.engine_config.forward_only=true' in source


def test_task_runtime_infers_strategy_without_direct_actor_attr_reads():
    source = _read_text("verl_gr/recipes/task_runtime.py")
    assert "def _infer_role_strategy(" in source
    assert "def _ensure_role_strategy(" in source
    assert "actor_strategy = self._ensure_role_strategy(config, \"actor\")" in source
    assert "config.actor_rollout_ref.actor.strategy in {\"fsdp\", \"fsdp2\", \"ddp\"}" not in source


def test_minionerec_worker_does_not_require_direct_actor_strategy_attr():
    source = _read_text("verl_gr/recipes/minionerec/minionerec_fsdp_workers.py")
    assert "actor_strategy =" in source
    assert "self.config.actor.strategy not in" not in source


def test_openonerec_yaml_has_stage2_decode_mode():
    source = _read_text("configs/verl_gr/openonerec/grpo_trainer.yaml")
    assert "stage2_decode_mode: vllm_native_beam" in source
