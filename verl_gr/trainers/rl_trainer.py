"""RL trainer runtime object for verl-GR."""

import os
import subprocess
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence

from verl_gr.integrations.runtime_adapter import TaskRuntime
from verl_gr.integrations.verl.ray_trainer import RayPPOTrainerRuntime


class OpenOneRecGRPORuntime(TaskRuntime):
    """Build and run OpenOneRec GRPO overrides using backend bridges."""

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        ray_runtime: RayPPOTrainerRuntime | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.local_recipe_root = self.project_root / "verl_gr" / "recipes" / "openonerec"
        self.ray_runtime = ray_runtime or RayPPOTrainerRuntime()
        self.last_overrides: list[str] = []
        self.last_command: list[str] = []
        self.status = "initialized"

    def _build_overrides(self, env: Mapping[str, str], n_nodes: int, n_gpus: int) -> list[str]:
        train_batch_size = int(env.get("TRAIN_BATCH_SIZE", str(n_nodes * n_gpus)))
        agent_loop_num_workers = int(env.get("AGENT_LOOP_NUM_WORKERS", "1"))
        fsdp_strategy = env.get("FSDP_STRATEGY", "fsdp")
        default_use_fused = "True" if fsdp_strategy == "fsdp2" else "False"
        default_fused_backend = "triton" if fsdp_strategy == "fsdp2" else "torch"
        use_fused_kernels = env.get("USE_FUSED_KERNELS", default_use_fused)
        use_remove_padding = env.get("USE_REMOVE_PADDING", "False")
        fused_kernel_impl_backend = env.get("FUSED_KERNEL_IMPL_BACKEND", default_fused_backend)
        onerec_recipe_path = self.local_recipe_root / "onerec_recipe.py"
        return [
            "algorithm.adv_estimator=grpo",
            f"data.train_files={env.get('TRAIN_FILES', '[]')}",
            f"data.val_files={env.get('VAL_FILES', '[]')}",
            "data.max_prompt_length=10240",
            f"++data.enable_think={env.get('ENABLE_THINK', 'False')}",
            f"++data.enable_nonthink={env.get('ENABLE_NONTHINK', 'False')}",
            f"++data.use_force_prefix={env.get('USE_FORCE_PREFIX', 'False')}",
            "data.prompt_key=prompt",
            "data.shuffle=True",
            f"data.max_response_length={env.get('RESPONSE_LENGTH', '2048')}",
            f"data.train_batch_size={train_batch_size}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            f"data.custom_cls.path={onerec_recipe_path}",
            "data.custom_cls.name=OneRecDataset",
            "data.reward_fn_key=source",
            "++data.data_source_key=source",
            f"custom_reward_function.path={onerec_recipe_path}",
            "custom_reward_function.name=compute_score",
            "actor_rollout_ref.ref.entropy_from_logits_with_chunking=True",
            "actor_rollout_ref.actor.entropy_checkpointing=True",
            "actor_rollout_ref.rollout.enable_chunked_prefill=True",
            "actor_rollout_ref.rollout.calculate_log_probs=False",
            "actor_rollout_ref.actor.clip_ratio_high=0.28",
            "actor_rollout_ref.model.enable_activation_offload=True",
            f"actor_rollout_ref.model.use_remove_padding={use_remove_padding}",
            f"actor_rollout_ref.actor.use_dynamic_bsz={env.get('USE_DYNAMIC_BSZ', 'True')}",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={env.get('MAX_TOKENS_PER_GPU', '40960')}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch_size}",
            f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={env.get('MAX_TOKENS_PER_GPU', '40960')}",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={env.get('MAX_TOKENS_PER_GPU', '40960')}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={env.get('MAX_TOKENS_PER_GPU', '40960')}",
            "actor_rollout_ref.rollout.max_num_seqs=2048",
            f"actor_rollout_ref.rollout.agent.num_workers={agent_loop_num_workers}",
            f"actor_rollout_ref.actor.optim.lr={env.get('LEARNING_RATE', '2e-6')}",
            "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
            "actor_rollout_ref.actor.optim.weight_decay=0.1",
            f"actor_rollout_ref.model.use_fused_kernels={use_fused_kernels}",
            f"actor_rollout_ref.model.fused_kernel_options.impl_backend={fused_kernel_impl_backend}",
            f"actor_rollout_ref.model.path={env.get('BASE_MODEL', '/path/to/your/model')}",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"actor_rollout_ref.rollout.n={env.get('ROLLOUT_N', '1')}",
            "actor_rollout_ref.rollout.dtype=bfloat16",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={env.get('ROLLOUT_TP_SIZE', '1')}",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.8",
            "++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=320",
            f"actor_rollout_ref.rollout.temperature={env.get('TEMPERATURE', '1')}",
            "actor_rollout_ref.rollout.top_p=1.0",
            "actor_rollout_ref.rollout.do_sample=True",
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={env.get('KL_LOSS_COEF', '0.001')}",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            "algorithm.norm_adv_by_std_in_grpo=True",
            "algorithm.use_kl_in_reward=False",
            "trainer.default_hdfs_dir=null",
            f"trainer.n_gpus_per_node={n_gpus}",
            f"trainer.nnodes={n_nodes}",
            "trainer.save_freq=50",
            "trainer.test_freq=50",
            f"trainer.project_name={env.get('PROJECT_NAME', 'OneRec_RL')}",
            f"trainer.experiment_name={env.get('EXPERIMENT_NAME', 'grpo_two_stage')}",
            f"trainer.default_local_dir={env.get('OUTPUT_DIR', 'outputs/openonerec')}/ckpt",
            "trainer.total_epochs=20",
            "trainer.val_before_train=True",
            f"actor_rollout_ref.ref.strategy={fsdp_strategy}",
            f"actor_rollout_ref.actor.strategy={fsdp_strategy}",
            "++critic.enable=False",
            "++actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer]",
            "++actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer]",
            "++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
            "++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        ]

    def set_run_mode(self, run_mode: str) -> None:
        self.ray_runtime.set_run_mode(run_mode)

    def run(
        self,
        env: Mapping[str, str],
        extra_overrides: Sequence[str] | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        self.status = "running"
        if not self.local_recipe_root.exists():
            raise FileNotFoundError(f"Local OpenOneRec recipe runtime not found at {self.local_recipe_root}")

        n_nodes, n_gpus = self.ray_runtime.detect_cluster()
        overrides = self._build_overrides(env, n_nodes=n_nodes, n_gpus=n_gpus)
        if extra_overrides:
            overrides.extend(extra_overrides)
        self.last_overrides = overrides

        runtime_env = dict(env)
        runtime_env["PYTHONPATH"] = f"{self.project_root}{os.pathsep}{runtime_env.get('PYTHONPATH', '')}"
        completed = self.ray_runtime.run_entrypoint(
            entrypoint_module="verl_gr.recipes.openonerec.main_onerec_ppo",
            overrides=overrides,
            cwd=self.project_root,
            env=runtime_env,
        )
        self.last_command = list(self.ray_runtime.last_command)
        self.status = self.ray_runtime.status
        return completed


class RLTrainer:
    """Stateful RL runtime orchestrator.

    The trainer can execute RL in two ways:
    - delegate to an injected runtime bridge (integration path)
    - launch a real OpenOneRec GRPO workload via script
    """

    def __init__(
        self,
        name: str = "rl_trainer",
        metadata: Mapping[str, Any] | None = None,
        runtime_bridge: Any | None = None,
        *,
        workload: str = "bridge",
        grpo_script_path: str | Path = "scripts/run_openonerec_grpo.sh",
        run_mode: str = "dry_run",
        grpo_runtime: TaskRuntime | None = None,
    ) -> None:
        self.name = name
        self.metadata = dict(metadata or {})
        self.runtime_bridge = runtime_bridge
        self.workload = workload
        self.grpo_script_path = Path(grpo_script_path)
        self.run_mode = run_mode
        self.grpo_runtime = grpo_runtime
        self.status = "initialized"
        self.last_runtime_args: dict[str, str] = {}
        self.last_result: RLOutput | None = None
        self.run_count = 0

    def validate_input(self, rl_input: RLInput) -> None:
        """Validate the minimal RL input contract."""

        if not rl_input.tokenized_samples:
            raise ValueError("RLInput.tokenized_samples must not be empty.")

    def _build_grpo_env(self, rl_input: RLInput) -> dict[str, str]:
        """Map contract values to GRPO launcher environment variables."""

        first_metadata = rl_input.tokenized_samples[0].metadata if rl_input.tokenized_samples else {}
        env = os.environ.copy()
        env.update(
            {
                "BASE_MODEL": str(rl_input.policy_model_path),
                "REF_MODEL": str(rl_input.reference_model_path) if rl_input.reference_model_path else "",
                "ROLLOUT_N": str(first_metadata.get("rollout_n", 1)),
                "STAGE1_MAX_TOKENS": str(first_metadata.get("stage1_max_tokens", 1024)),
                "STAGE2_BEAM_SIZE": str(first_metadata.get("stage2_beam_size", 32)),
                "STAGE2_NUM_TOKENS": str(first_metadata.get("stage2_num_tokens", 3)),
                "ROLLOUT_TP_SIZE": str(first_metadata.get("rollout_tp_size", 1)),
                "TEMPERATURE": str(first_metadata.get("temperature", 1.0)),
                "USE_DYNAMIC_BSZ": str(first_metadata.get("use_dynamic_bsz", "True")),
                "MAX_TOKENS_PER_GPU": str(first_metadata.get("max_tokens_per_gpu", 40960)),
                "TRAIN_FILES": str(first_metadata.get("train_files", "[]")),
                "VAL_FILES": str(first_metadata.get("val_files", "[]")),
                "OUTPUT_DIR": str(first_metadata.get("output_dir", "outputs/openonerec")),
                "PROJECT_NAME": str(first_metadata.get("project_name", "verl-GR")),
                "EXPERIMENT_NAME": str(first_metadata.get("experiment_name", "grpo_two_stage")),
                "ENABLE_THINK": str(first_metadata.get("enable_think", "False")),
                "ENABLE_NONTHINK": str(first_metadata.get("enable_nonthink", "False")),
                "USE_FORCE_PREFIX": str(first_metadata.get("use_force_prefix", "False")),
            }
        )
        return env

    def _run_grpo_workload(self, rl_input: RLInput) -> RLOutput:
        """Launch OpenOneRec GRPO workload via RayPPO runtime or script fallback."""

        runtime_env = self._build_grpo_env(rl_input)
        self.last_runtime_args = {
            key: runtime_env[key]
            for key in ("BASE_MODEL", "ROLLOUT_N", "STAGE2_BEAM_SIZE", "OUTPUT_DIR")
        }

        runtime = self.grpo_runtime or OpenOneRecGRPORuntime()
        runtime.set_run_mode(self.run_mode)
        try:
            completed = runtime.run(runtime_env)
            if completed is None:
                return RLOutput(
                    checkpoint=CheckpointArtifact(
                        stage_name="rl",
                        checkpoint_root=Path(runtime_env["OUTPUT_DIR"]) / "ckpt",
                        metadata={
                            "mode": "dry_run",
                            "workload": "grpo",
                            "runner": "OpenOneRecGRPORuntime",
                        },
                    ),
                    metrics={"status": "initialized", "run_mode": self.run_mode},
                    traces={"command": " ".join(runtime.last_command)},
                )

            return RLOutput(
                checkpoint=CheckpointArtifact(
                    stage_name="rl",
                    checkpoint_root=Path(runtime_env["OUTPUT_DIR"]) / "ckpt",
                    metadata={
                        "mode": "execute",
                        "workload": "grpo",
                        "runner": "OpenOneRecGRPORuntime",
                    },
                ),
                metrics={"status": "completed", "returncode": completed.returncode},
                traces={
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                },
            )
        except FileNotFoundError:
            # fallback to script launcher when OpenOneRec runtime path is unavailable
            script_path = self.grpo_script_path
            if not script_path.is_absolute():
                script_path = Path(__file__).resolve().parents[2] / script_path
            if not script_path.exists():
                raise

        if self.run_mode != "execute":
            return RLOutput(
                checkpoint=CheckpointArtifact(
                    stage_name="rl",
                    checkpoint_root=Path(runtime_env["OUTPUT_DIR"]) / "ckpt",
                    metadata={
                        "mode": "dry_run",
                        "workload": "grpo",
                        "script_path": str(script_path),
                        "runner": "script_fallback",
                    },
                ),
                metrics={"status": "initialized", "run_mode": self.run_mode},
                traces={"command": f"bash {script_path}"},
            )

        completed = subprocess.run(
            ["bash", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
            env=runtime_env,
        )
        return RLOutput(
            checkpoint=CheckpointArtifact(
                stage_name="rl",
                checkpoint_root=Path(runtime_env["OUTPUT_DIR"]) / "ckpt",
                metadata={
                    "mode": "execute",
                    "workload": "grpo",
                    "script_path": str(script_path),
                        "runner": "script_fallback",
                },
            ),
            metrics={"status": "completed", "returncode": completed.returncode},
            traces={
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
        )

    def run(self, rl_input: RLInput) -> RLOutput:
        """Execute RL training runtime."""

        self.validate_input(rl_input)
        self.status = "running"
        self.run_count += 1

        result: RLOutput
        if self.workload == "grpo":
            result = self._run_grpo_workload(rl_input)
        elif self.runtime_bridge is not None and hasattr(self.runtime_bridge, "run"):
            result = self.runtime_bridge.run(rl_input)
        else:
            raise NotImplementedError("RLTrainer needs a runtime bridge or workload='grpo'.")

        self.last_result = result
        self.status = "completed"
        return result


# -----------------------------
# OneRec custom ray-trainer API
# -----------------------------
ray_trainer_mod = import_module("verl.trainer.ppo.ray_trainer")
metric_utils_mod = import_module("verl.trainer.ppo.metric_utils")
core_algos = import_module("verl.trainer.ppo.core_algos")
protocol_mod = import_module("verl.protocol")
torch_functional = import_module("verl.utils.torch_functional")
reward_mod = import_module("verl.trainer.ppo.reward")
DataProto = getattr(import_module("verl"), "DataProto")
np = import_module("numpy")
torch = import_module("torch")

Role = getattr(ray_trainer_mod, "Role")
ResourcePoolManager = getattr(ray_trainer_mod, "ResourcePoolManager")
AdvantageEstimator = getattr(core_algos, "AdvantageEstimator")

process_validation_metrics = getattr(metric_utils_mod, "process_validation_metrics")
pad_dataproto_to_divisor = getattr(protocol_mod, "pad_dataproto_to_divisor")
unpad_dataproto = getattr(protocol_mod, "unpad_dataproto")
masked_mean = getattr(torch_functional, "masked_mean")
extract_reward = getattr(reward_mod, "extract_reward")
RayPPOTrainerBase = getattr(ray_trainer_mod, "RayPPOTrainer")


def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty: str = "kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    return data, {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,  # noqa: ARG001
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    tokenizer=None,  # noqa: ARG001
) -> DataProto:
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer(RayPPOTrainerBase):
    """RayPPOTrainer override with OneRec validation/beam expansion handling."""

    @staticmethod
    def _ensure_reward_routing_keys(proto: DataProto) -> None:
        """Ensure both source aliases exist for reward-loop compatibility."""
        non_tensor = proto.non_tensor_batch
        if "data_source" not in non_tensor and "source" in non_tensor:
            non_tensor["data_source"] = non_tensor["source"]
        if "source" not in non_tensor and "data_source" in non_tensor:
            non_tensor["source"] = non_tensor["data_source"]

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Prepare generation batch without conflicting prompt tensors.

        In verl>=0.7.1 async rollout mode, generation output may include input_ids.
        If original training batch still carries prompt-side input_ids/attention_mask/
        position_ids, DataProto.union() asserts on key collisions. For OneRec dataset,
        we remove those prompt tensors before generation and keep reward-routing keys.
        """
        reward_keys = set({"source", "data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = [
            key for key in ("input_ids", "attention_mask", "position_ids") if key in batch.batch.keys()
        ]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        self._ensure_reward_routing_keys(gen_batch)
        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []
        sample_ground_truths = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
            rollout_config = self.config.actor_rollout_ref.rollout
            use_beam_search_val = val_kwargs.get("use_beam_search", False)
            is_two_stage_rollout_val = rollout_config.get("name") == "two_stage"

            if not use_beam_search_val:
                test_batch = test_batch.repeat(repeat_times=val_kwargs.n, interleave=True)

            if (
                self.use_rm
                and "reward_model" in test_batch[0].non_tensor_batch
                and test_batch[0].non_tensor_batch["reward_model"].get("style") == "model"
            ):
                return {}

            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            if "reward_model" in test_batch.non_tensor_batch:
                ground_truths = [item["ground_truth"] for item in test_batch.non_tensor_batch["reward_model"]]
            else:
                ground_truths = []

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            for key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "interaction_kwargs", "agent_name"):
                if key in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append(key)
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            # Keep reward-routing metadata in generation batch so async reward loop
            # can resolve source-specific scoring during validation.
            for key in ("source", "data_source", "reward_model", "extra_info", "uid"):
                if key in test_batch.non_tensor_batch and key not in test_gen_batch.non_tensor_batch:
                    test_gen_batch.non_tensor_batch[key] = test_batch.non_tensor_batch[key]
            self._ensure_reward_routing_keys(test_gen_batch)

            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            if is_two_stage_rollout_val:
                meta_info["enable_two_stage_rollout"] = True
                meta_info["stage1_max_tokens"] = rollout_config.get(
                    "stage1_max_tokens",
                    self.config.data.get("max_response_length", 1024),
                )
                meta_info["stage2_beam_size"] = rollout_config.get("stage2_beam_size", 32)
                meta_info["stage2_num_tokens"] = rollout_config.get("stage2_num_tokens", 3)
                meta_info["max_tokens"] = self.config.data.get("max_response_length", 1024)
                meta_info["use_beam_search"] = False
                meta_info["n"] = val_kwargs.get("n", 1)
                meta_info["return_all_beams"] = True
            elif use_beam_search_val:
                meta_info["use_beam_search"] = True
                meta_info["best_of"] = val_kwargs.get("best_of", 4)
                meta_info["max_tokens"] = self.config.data.get("max_response_length", 16)
                meta_info["temperature"] = 0
                meta_info["n"] = val_kwargs.get("n", 1)
                meta_info["return_all_beams"] = True

            test_gen_batch.meta_info = meta_info
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if use_beam_search_val or is_two_stage_rollout_val:
                n_beams = (
                    rollout_config.get("stage2_beam_size", 2)
                    if is_two_stage_rollout_val
                    else val_kwargs.get("n", 1)
                )
                actual_pad_size = pad_size * n_beams
            else:
                actual_pad_size = pad_size
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=actual_pad_size)

            output_len = len(test_output_gen_batch)
            input_len = len(test_batch)
            if output_len > input_len and (use_beam_search_val or is_two_stage_rollout_val):
                expand_factor = output_len // input_len
                test_batch = test_batch.repeat(repeat_times=expand_factor, interleave=True)
                input_texts = [t for t in input_texts for _ in range(expand_factor)]
                if ground_truths:
                    ground_truths = [t for t in ground_truths for _ in range(expand_factor)]

            sample_inputs.extend(input_texts)
            if ground_truths:
                sample_ground_truths.extend(ground_truths)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            response_lengths = [(ids != self.tokenizer.pad_token_id).sum().item() for ids in output_ids]
            reward_extra_infos_dict["response_length"].extend(response_lengths)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            if "generated_items" in test_batch.non_tensor_batch:
                generated_items_arr = test_batch.non_tensor_batch["generated_items"]
                batch_size = len(generated_items_arr)
                if "extra_info" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(batch_size)], dtype=object)
                extra_info_arr = test_batch.non_tensor_batch["extra_info"]
                for i in range(batch_size):
                    if extra_info_arr[i] is None:
                        extra_info_arr[i] = {}
                    extra_info_arr[i]["generated_items"] = generated_items_arr[i]

            reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                elif isinstance(values, list):
                    reward_extra_infos_dict[key].extend(values)
                else:
                    reward_extra_infos_dict[key].append(values)

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            reward_fn_key = self.config.data.get("reward_fn_key", "data_source")
            data_sources_batch = test_batch.non_tensor_batch.get(reward_fn_key)
            if data_sources_batch is None:
                data_sources_batch = test_batch.non_tensor_batch.get("source")
            if data_sources_batch is None:
                data_sources_batch = test_batch.non_tensor_batch.get("data_source")
            if data_sources_batch is None:
                data_sources_batch = ["unknown"] * reward_tensor.shape[0]
            data_source_lst.append(data_sources_batch)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                ground_truths=sample_ground_truths,
            )

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys())
                for metric_name, metric_val in metric2val.items():
                    is_core = (
                        var_name == core_var
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best", "pass"])
                        and f"@{n_max}" in metric_name
                    )
                    metric_sec = "val-core" if is_core else "val-aux"
                    metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if "response_length" in reward_extra_infos_dict and len(reward_extra_infos_dict["response_length"]) > 0:
            response_lengths_tensor = torch.tensor(reward_extra_infos_dict["response_length"])
            metric_dict["val/response_length/mean"] = response_lengths_tensor.float().mean().item()
            metric_dict["val/response_length/max"] = response_lengths_tensor.max().item()
            metric_dict["val/response_length/min"] = response_lengths_tensor.min().item()
        return metric_dict

