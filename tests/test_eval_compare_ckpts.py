import ast
from pathlib import Path
from types import SimpleNamespace


def _load_helper(source: str, helper_name: str):
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == helper_name:
            namespace = {"np": SimpleNamespace(number=(int, float))}
            helper_module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(helper_module)
            exec(compile(helper_module, "<eval_compare_ckpts_helper>", "exec"), namespace)
            return namespace[helper_name]
    raise AssertionError(f"{helper_name} not found")


def test_eval_compare_metric_helper_averages_per_sample_lists():
    source = Path("scripts/eval_compare_ckpts.py").read_text(encoding="utf-8")
    mean_metric_value = _load_helper(source, "_mean_metric_value")

    assert mean_metric_value([1.0, 0.0, 1.0]) == 2.0 / 3.0
    assert mean_metric_value([0.5]) == 0.5
    assert mean_metric_value([]) == 0.0
    assert mean_metric_value(0.25) == 0.25


def test_eval_compare_table_uses_metric_means_not_first_sample():
    source = Path("scripts/eval_compare_ckpts.py").read_text(encoding="utf-8")

    assert "_mean_metric_value(results1.get(k, []))" in source
    assert "_mean_metric_value(results2.get(k, []))" in source
    assert "results1.get(k, [0])[0]" not in source
    assert "results2.get(k, [0])[0]" not in source
