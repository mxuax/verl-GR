import ast
from pathlib import Path


def test_two_stage_rollout_enables_raw_prompt_schema():
    source = Path("verl_gr/recipes/openonerec/onerec_recipe.py").read_text()
    module = ast.parse(source)

    configure_rollout = None
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "OneRecTask":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "configure_rollout":
                    configure_rollout = item
                    break
    assert configure_rollout is not None

    for node in ast.walk(configure_rollout):
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
            and args[1].value == "data.return_raw_chat"
            and isinstance(args[2], ast.Constant)
            and args[2].value is True
            and any(keyword.arg == "force_add" and keyword.value.value is True for keyword in node.keywords)
        ):
            break
    else:
        raise AssertionError("two_stage rollout must enable data.return_raw_chat for agent raw_prompt input")
