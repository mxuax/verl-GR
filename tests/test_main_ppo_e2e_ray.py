from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
torch = pytest.importorskip("torch")
pytest.importorskip("hydra")
pytest.importorskip("verl")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _DummyActorRolloutWorker:
    pass


class _DummyCriticWorker:
    pass


class _DummyRewardWorker:
    pass


class _DummyRayWorkerGroup:
    pass


class _DummyPrompts:
    def __init__(self):
        self.batch = {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
        self.meta_info = {"eos_token_id": 2}


class _DummyPreparedInputs:
    def __init__(self):
        self.vllm_inputs = [{"prompt_token_ids": [11, 12]}]
        self.non_tensor_batch = {}


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        _ = (text, add_special_tokens)
        return [97]

    def __len__(self):
        return 1000


class _DummyInferenceEngine:
    def __init__(self):
        self._tokenizer = _DummyTokenizer()

    def generate(self, prompts, sampling_params, lora_request, use_tqdm=False):
        _ = (prompts, sampling_params, lora_request, use_tqdm)
        token_output = type("TokenOutput", (), {"token_ids": [13, 14]})()
        return [type("GenerationOutput", (), {"outputs": [token_output]})()]

    def beam_search(self, prompts, params):
        _ = params
        return [{"prompt": prompts[0], "beam_token_ids": [[21, 22], [23]]}]

    def get_tokenizer(self):
        return self._tokenizer


class _FakeBeamSearchParams:
    def __init__(self, beam_width, max_tokens):
        self.beam_width = beam_width
        self.max_tokens = max_tokens


class _FakeExpansion:
    def __init__(self):
        self.idx = torch.tensor([[11, 12]], dtype=torch.long)
        self.responses = torch.tensor([[21, 22]], dtype=torch.long)
        self.attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        self.position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        self.non_tensor_batch = {}


def _exercise_two_stage_rollout(config):
    rollout_module = importlib.import_module("verl_gr.workers.rollout.two_stage_vllm_rollout")
    rollout = rollout_module.TwoStagevLLMRollout.__new__(rollout_module.TwoStagevLLMRollout)
    rollout.pad_token_id = 0
    rollout.config = OmegaConf.create(
        {"response_length": 8, "calculate_log_probs": False, "custom": {}}
    )
    rollout.inference_engine = _DummyInferenceEngine()
    rollout.lora_kwargs = {}

    prompts = _DummyPrompts()
    result = rollout._two_stage_generation(
        prompts,
        max_tokens=4,
        stage2_beam_size=2,
        stage2_num_tokens=3,
    )
    assert result["packed"] is True
    ray.get(config.test_state_actor.set.remote("two_stage_called", True))


class _FakeResourcePoolManager:
    def __init__(self, resource_pool_spec, mapping):
        self.resource_pool_spec = resource_pool_spec
        self.mapping = mapping


class _FakeRLTrainer:
    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self._state_actor = self._config.test_state_actor
        ray.get(self._state_actor.set.remote("trainer_ctor_called", True))

    def init_workers(self):
        ray.get(self._state_actor.set.remote("init_workers_called", True))

    def fit(self):
        _exercise_two_stage_rollout(self._config)
        ray.get(self._state_actor.set.remote("fit_called", True))


class _FakeOneRecTask:
    def sanitize_fsdp2_wrap_policy(self, config):
        ray.get(config.test_state_actor.inc.remote("sanitize_calls"))

    def prepare(self, config):
        ray.get(config.test_state_actor.set.remote("prepare_called", True))
        return {
            "tokenizer": "tokenizer",
            "processor": "processor",
            "actor_rollout_cls": _DummyActorRolloutWorker,
            "critic_worker": _DummyCriticWorker,
            "reward_model_worker": _DummyRewardWorker,
            "reward_model_cfg": {"enable": False},
            "ray_worker_group_cls": _DummyRayWorkerGroup,
        }


def _build_config(state_actor):
    return OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
            "critic": {"enable": True},
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            "data": {"train_files": ["train.jsonl"], "val_files": ["val.jsonl"]},
            "test_state_actor": state_actor,
        },
        flags={"allow_objects": True},
    )


@ray.remote
class _StateRecorder:
    def __init__(self):
        self._values = {
            "sanitize_calls": 0,
            "auto_set_device_called": False,
            "migrate_called": False,
            "prepare_called": False,
            "trainer_ctor_called": False,
            "init_workers_called": False,
            "fit_called": False,
            "datasets_seen": [],
            "sampler_called": False,
            "two_stage_called": False,
        }

    def inc(self, key):
        self._values[key] += 1

    def set(self, key, value=True):
        self._values[key] = value

    def append(self, key, value):
        self._values[key].append(value)

    def snapshot(self):
        return self._values


@pytest.fixture
def ray_runtime():
    if ray.is_initialized():
        yield
        return
    ray.init(local_mode=True, num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        yield
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_main_ppo_end_to_end_runtime_with_ray(monkeypatch, ray_runtime):
    module = importlib.import_module("verl_gr.trainers.main_ppo")
    rollout_module = importlib.import_module("verl_gr.workers.rollout.two_stage_vllm_rollout")
    state_actor = _StateRecorder.remote()
    config = _build_config(state_actor)

    monkeypatch.setattr(module, "OneRecTask", _FakeOneRecTask)
    monkeypatch.setattr(module, "ResourcePoolManager", _FakeResourcePoolManager)
    monkeypatch.setattr(module, "RLTrainer", _FakeRLTrainer)

    def _fake_auto_set_device(cfg):
        ray.get(cfg.test_state_actor.set.remote("auto_set_device_called", True))

    def _fake_migrate(cfg):
        ray.get(cfg.test_state_actor.set.remote("migrate_called", True))
        return cfg

    def _fake_dataset(files, data_cfg, tokenizer, processor, is_train):
        ray.get(
            config.test_state_actor.append.remote(
                "datasets_seen",
                ("train" if is_train else "val", tuple(files)),
            )
        )
        return [{"kind": "train" if is_train else "val"}]

    def _fake_sampler(data_cfg, train_dataset):
        ray.get(config.test_state_actor.set.remote("sampler_called", True))
        return "sampler"

    def _fake_base_run_ppo(cfg, task_runner_class):
        assert ray.is_initialized(), "Ray must be initialized for this end-to-end test."
        runner = task_runner_class.remote()
        ray.get(runner.run.remote(cfg))

    monkeypatch.setattr(module, "auto_set_device", _fake_auto_set_device)
    monkeypatch.setattr(module, "migrate_legacy_reward_impl", _fake_migrate)
    monkeypatch.setattr(module, "create_rl_dataset", _fake_dataset)
    monkeypatch.setattr(module, "create_rl_sampler", _fake_sampler)
    monkeypatch.setattr(module, "base_run_ppo", _fake_base_run_ppo)
    monkeypatch.setattr(rollout_module, "prepare_prompt_token_inputs", lambda *args, **kwargs: _DummyPreparedInputs())
    monkeypatch.setattr(rollout_module, "build_sampling_params", lambda **kwargs: kwargs)
    monkeypatch.setattr(rollout_module, "build_lora_requests", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_module, "BeamSearchParams", _FakeBeamSearchParams)
    monkeypatch.setattr(rollout_module, "expand_beam_candidates", lambda **kwargs: _FakeExpansion())
    monkeypatch.setattr(
        rollout_module,
        "pack_rollout_batch",
        lambda **kwargs: {"packed": True, "payload": kwargs},
    )

    hydra_main = module._build_main()
    hydra_main.__wrapped__(config)

    state = ray.get(state_actor.snapshot.remote())
    assert state["auto_set_device_called"]
    assert state["migrate_called"]
    assert state["prepare_called"]
    assert state["trainer_ctor_called"]
    assert state["init_workers_called"]
    assert state["fit_called"]
    assert state["sampler_called"]
    assert state["two_stage_called"]
    assert ("train", ("train.jsonl",)) in state["datasets_seen"]
    assert ("val", ("val.jsonl",)) in state["datasets_seen"]
    assert state["sanitize_calls"] >= 2
