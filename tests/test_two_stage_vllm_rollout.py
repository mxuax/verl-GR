import asyncio
import importlib
import sys
import types
import unittest


class _NoGrad:
    def __call__(self, func=None):
        if func is None:
            return self
        return func


class _FakeRemoteMethod:
    def __init__(self, calls, name):
        self._calls = calls
        self._name = name

    async def remote(self, **kwargs):
        self._calls.append((self._name, kwargs))
        return {"method": self._name, "kwargs": kwargs}


class _FakeActor:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return _FakeRemoteMethod(self.calls, name)


class _FakeRay(types.ModuleType):
    def __init__(self):
        super().__init__("ray")
        self.actor_names = []
        self.actor = _FakeActor()

    def get_actor(self, name):
        self.actor_names.append(name)
        return self.actor


def _install_import_stubs():
    fake_ray = _FakeRay()
    sys.modules["ray"] = fake_ray

    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = _NoGrad
    sys.modules["torch"] = torch_mod

    verl_mod = types.ModuleType("verl")
    verl_mod.DataProto = type("DataProto", (), {})
    sys.modules["verl"] = verl_mod

    third_party_vllm = types.ModuleType("verl_gr.third_party.vllm")
    third_party_vllm.BeamSearchParams = None
    third_party_vllm.LoRARequest = type("LoRARequest", (), {})
    sys.modules["verl_gr.third_party.vllm"] = third_party_vllm

    class _ServerAdapter:
        def __init__(self, *args, **kwargs):
            pass

        async def update_weights(self, weights, global_steps=None, **kwargs):
            return {"updated": True}

    vllm_rollout_mod = types.ModuleType("verl.workers.rollout.vllm_rollout.vllm_rollout")
    vllm_rollout_mod.ServerAdapter = _ServerAdapter
    sys.modules["verl.workers.rollout.vllm_rollout.vllm_rollout"] = vllm_rollout_mod

    for name in [
        "verl.workers",
        "verl.workers.rollout",
        "verl.workers.rollout.vllm_rollout",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))

    primitives = types.ModuleType("verl_gr.workers.rollout.primitives")
    for attr in [
        "build_lora_requests",
        "build_sampling_params",
        "expand_beam_candidates",
        "pack_rollout_batch",
        "prepare_prompt_token_inputs",
    ]:
        setattr(primitives, attr, lambda *args, **kwargs: None)
    sys.modules["verl_gr.workers.rollout.primitives"] = primitives

    return fake_ray


class TwoStageRolloutServerMethodTest(unittest.TestCase):
    def setUp(self):
        self.fake_ray = _install_import_stubs()
        sys.modules.pop("verl_gr.workers.rollout.two_stage_vllm_rollout", None)
        self.module = importlib.import_module("verl_gr.workers.rollout.two_stage_vllm_rollout")

    def test_execute_server_method_uses_upstream_vllm_actor_name(self):
        rollout = object.__new__(self.module.TwoStagevLLMRollout)
        rollout.rollout_rank = 0
        rollout.server_handle = None
        rollout.replica_rank = 3
        rollout.node_rank = 2

        result = asyncio.run(rollout._execute_server_method("abort_all_requests", reset_prefix_cache=True))

        self.assertEqual(self.fake_ray.actor_names, ["vllm_server_3_2"])
        self.assertEqual(result, {"method": "abort_all_requests", "kwargs": {"reset_prefix_cache": True}})
        self.assertEqual(
            self.fake_ray.actor.calls,
            [("abort_all_requests", {"reset_prefix_cache": True})],
        )

    def test_execute_server_method_skips_nonzero_rollout_rank(self):
        rollout = object.__new__(self.module.TwoStagevLLMRollout)
        rollout.rollout_rank = 1
        rollout.server_handle = None
        rollout.replica_rank = 3
        rollout.node_rank = 2

        result = asyncio.run(rollout._execute_server_method("abort_all_requests", reset_prefix_cache=True))

        self.assertIsNone(result)
        self.assertEqual(self.fake_ray.actor_names, [])


if __name__ == "__main__":
    unittest.main()
