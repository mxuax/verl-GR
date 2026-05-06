import ast
from pathlib import Path


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
    source = Path("verl_gr/trainers/rl_trainer.py").read_text()
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
    source = Path("verl_gr/trainers/rl_trainer.py").read_text()
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
    source = Path("verl_gr/recipes/minionerec/minionerec_trainer.py").read_text()
    module = ast.parse(source)
    adapter_cls = _get_class(module, "MiniOneRecTrainerAdapter")
    method = _get_method(adapter_cls, "_compute_ranking_metrics")

    source_segment = ast.get_source_segment(source, method) or ""
    assert "sample_uids" in source_segment
    assert "grouped_indices[(str(data_source), str(uid))]" in source_segment
    assert "sample_inputs" not in source_segment


def test_minionerec_dataset_disables_alignment_for_val_by_default():
    source = Path("verl_gr/recipes/minionerec/minionerec_dataset.py").read_text()
    module = ast.parse(source)
    dataset_cls = _get_class(module, "MiniOneRecDataset")
    init_method = _get_method(dataset_cls, "__init__")
    init_segment = ast.get_source_segment(source, init_method) or ""

    assert 'config.get("include_alignment_tasks_for_val", False)' in init_segment
    assert "self.is_val_split = self._is_val_split" in init_segment
    assert "self.include_alignment_tasks = (" in init_segment


def test_minionerec_agent_loop_supports_train_val_decode_modes():
    source = Path("verl_gr/recipes/minionerec/constrained_beam_agent_loop.py").read_text()
    module = ast.parse(source)
    worker_cls = _get_class(module, "MiniOneRecConstrainedBeamAgentLoopWorker")
    generate_fn = _get_method(worker_cls, "generate_sequences")
    segment = ast.get_source_segment(source, generate_fn) or ""

    assert 'rollout_custom.get("decode_mode_train", "stochastic_constrained")' in segment
    assert 'rollout_custom.get("decode_mode_val", "deterministic_beam")' in segment
    assert 'sampling_params[BEAM_SEARCH_PARAMS_KEY]["decode_mode"] = decode_mode' in segment
    assert 'sampling_params[BEAM_SEARCH_PARAMS_KEY]["disable_cache_in_train"]' in segment


def test_constrained_beam_server_has_stochastic_mode_branch():
    source = Path("verl_gr/workers/rollout/constrained_beam_vllm_async.py").read_text()
    assert 'beam_config.decode_mode == "stochastic_constrained"' in source
    assert "_run_constrained_stochastic_sample" in source
