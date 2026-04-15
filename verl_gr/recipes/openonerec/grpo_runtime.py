"""OpenOneRec GRPO runtime assembly owned by recipe layer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

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
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self.local_recipe_root = self.project_root / "verl_gr" / "recipes" / "openonerec"
        self.ray_runtime = ray_runtime or RayPPOTrainerRuntime()
        self.last_overrides: list[str] = []
        self.last_command: list[str] = []
        self.status = "initialized"

    def _build_overrides(self, env: Mapping[str, str], n_nodes: int, n_gpus: int) -> list[str]:
        train_batch_size = int(env.get("TRAIN_BATCH_SIZE", str(n_nodes * n_gpus)))
        agent_loop_num_workers = int(env.get("AGENT_LOOP_NUM_WORKERS", "1"))
        use_fused_kernels = env.get("USE_FUSED_KERNELS", "False")
        use_remove_padding = env.get("USE_REMOVE_PADDING", "False")
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
            "actor_rollout_ref.ref.strategy=fsdp2",
            "actor_rollout_ref.actor.strategy=fsdp2",
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

