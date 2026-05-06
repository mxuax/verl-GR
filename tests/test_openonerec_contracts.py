import ast
from pathlib import Path


def test_onerec_task_configure_rollout_sets_default_agent_loop():
    source = Path("verl_gr/recipes/openonerec/onerec_recipe.py").read_text()
    module = ast.parse(source)

    configure_rollout_fn = None
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "OneRecTask":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "configure_rollout":
                    configure_rollout_fn = item
                    break
    assert configure_rollout_fn is not None

    for node in ast.walk(configure_rollout_fn):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "OmegaConf"
        ):
            continue
        args = node.args
        if (
            len(args) >= 3
            and isinstance(args[1], ast.Constant)
            and args[1].value == "actor_rollout_ref.rollout.agent.default_agent_loop"
            and isinstance(args[2], ast.Constant)
            and args[2].value == "openonerec_two_stage_agent"
            and any(keyword.arg == "force_add" and keyword.value.value is True for keyword in node.keywords)
        ):
            break
    else:
        raise AssertionError("OneRecTask.configure_rollout must set two-stage default agent loop")


def test_openonerec_trainer_adapter_keeps_legacy_validate_and_checkpoint_helpers():
    source = Path("verl_gr/recipes/openonerec/onerec_trainer.py").read_text()
    module = ast.parse(source)

    adapter_cls = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenOneRecTrainerAdapter":
            adapter_cls = node
            break
    assert adapter_cls is not None, "OpenOneRecTrainerAdapter should exist for adapter routing"

    method_names = {node.name for node in adapter_cls.body if isinstance(node, ast.FunctionDef)}
    assert "validate" in method_names
    assert "evaluate_and_prune_checkpoint" in method_names
