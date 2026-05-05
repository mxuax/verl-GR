import ast
from pathlib import Path


def _find_method(module, class_name, method_name):
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_rltrainer_validate_routes_through_task_adapter():
    source = Path("verl_gr/trainers/rl_trainer.py").read_text()
    module = ast.parse(source)
    validate_method = _find_method(module, "RLTrainer", "_validate")

    calls_adapter_validate = False
    for node in ast.walk(validate_method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "validate"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "_get_task_adapter"
        ):
            calls_adapter_validate = True

        assert not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "openonerec_validate"
        ), "RLTrainer._validate must not bypass task-specific validation"

    assert calls_adapter_validate, "RLTrainer._validate must dispatch to the task adapter"


def test_openonerec_default_adapter_forwards_legacy_validation_hooks():
    source = Path("verl_gr/recipes/openonerec/onerec_trainer.py").read_text()
    module = ast.parse(source)
    adapter = next(
        node for node in ast.walk(module) if isinstance(node, ast.ClassDef) and node.name == "OpenOneRecTrainerAdapter"
    )
    methods = {item.name: item for item in adapter.body if isinstance(item, ast.FunctionDef)}

    assert "validate" in methods
    assert "dump_generations" in methods
    assert "maybe_log_val_generations" in methods

    expected_calls = {
        "validate": "openonerec_validate",
        "dump_generations": "openonerec_dump_generations",
        "maybe_log_val_generations": "openonerec_maybe_log_val_generations",
    }
    for method_name, function_name in expected_calls.items():
        assert any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function_name
            for node in ast.walk(methods[method_name])
        ), f"{method_name} must forward to {function_name}"
