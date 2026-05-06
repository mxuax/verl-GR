import ast
from pathlib import Path


def test_task_registry_includes_minionerec():
    source = Path("verl_gr/trainers/main_ppo.py").read_text()
    assert '"minionerec": TaskSpec(name="minionerec", factory=MiniOneRecTask)' in source


def test_legacy_task_inference_maps_constrained_beam_to_minionerec():
    source = Path("verl_gr/trainers/main_ppo.py").read_text()
    module = ast.parse(source)
    infer_fn = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_infer_legacy_task_name":
            infer_fn = node
            break
    assert infer_fn is not None
    segment = ast.get_source_segment(source, infer_fn) or ""
    assert 'if rollout_name == "constrained_beam":' in segment
    assert 'return "minionerec"' in segment
