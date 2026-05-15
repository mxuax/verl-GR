"""MiniOneRec worker shim — vLLM-free: all generation routed through HF model.generate()."""

from __future__ import annotations

import contextlib

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl_gr.recipes.minionerec.minionerec_loss import (  # noqa: F401  # register REINFORCE loss on every worker
    compute_policy_loss_minionerec_reinforce,
)
from verl_gr.workers.ref_sync import RefSyncMixin
from verl_gr.workers.rollout.registration import register_constrained_beam_rollout_class


class MiniOneRecActorRolloutRefWorker(RefSyncMixin, ActorRolloutRefWorker):
    """FSDP worker that skips vLLM engine creation.

    MiniOneRec routes all generation (train + val) through
    HF ``model.generate()`` with FSDP ``summon_full_params``.
    vLLM is never used, so we prevent the parent ``init_model``
    from creating a local rollout engine and override
    ``update_weights`` to be a no-op for the naive checkpoint
    backend.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        actor_strategy = str(self.config.actor.get("strategy", "") or self.config.actor.get("engine_config", {}).get("strategy", "") or "").lower()
        # DDP skips registration: it uses hf_constrained_beam_generate directly
        # on the unwrapped module instead of the rollout-class dispatch path.
        if (self.config.rollout.name == "constrained_beam"
            and actor_strategy not in ("ddp",)):
            register_constrained_beam_rollout_class()

        # self.role is a str (e.g. "actor_rollout_ref"), not a set.
        # The parent init_model checks `"rollout" in self.role` via
        # substring match.  Temporarily replace the substring so that
        # the check fails and no vLLM engine is created.
        saved_role: str = self.role
        self.role = saved_role.replace("rollout", "no_rl")
        try:
            return super().init_model()
        finally:
            self.role = saved_role

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """Skip vLLM weight sync — MiniOneRec uses HF generate exclusively.

        The naive checkpoint-engine backend calls this on every FSDP worker
        after each actor update.  The original implementation resumes the
        colocated vLLM engine, syncs weights, and re-sleeps it.  MiniOneRec
        has no vLLM engine, so the naive path is a no-op.

        .. note::
           This override is only reached when
           ``checkpoint_engine.backend == "naive"`` (the default).  Non-naive
           backends would attempt to build a process group from
           ``rollout_replicas`` inside ``CheckpointEngineManager`` before
           dispatching, and would fail because ``_initialize_llm_servers``
           sets ``rollout_replicas = []``.
        """
        # Naive backend: no vLLM engine → nothing to sync.
        # (Non-naive backends don't reach here — they crash earlier inside
        #  CheckpointEngineManager.update_weights when replicas=[].)
        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def hf_constrained_beam_generate(self, prompts: list[str], meta_info: dict) -> dict:
        """Generate constrained beam completions on the HF (FSDP) model.

        ALL ranks receive the same prompt list (ONE_TO_ALL).  Each rank
        processes a round-robin shard to avoid duplicate work.  Results
        are returned per-rank; the trainer aggregates them.

        Args:
            prompts: full list of prompt strings (identical across ranks).
            meta_info: dict with ``beam_width``, ``do_sample``, ``info_file``,
                       ``temperature``, ``max_new_tokens``, ``validate``.

        Returns:
            dict with keys:
                ``prompt_indices``: global prompt indices handled by this rank.
                ``response_ids``: per-prompt grouped response IDs for this rank's shard.
        """
        from transformers import AutoTokenizer

        from verl_gr.recipes.minionerec.hf_constrained_generation import HfConstrainedBeamGenerator

        # Shard: each DP rank processes a round-robin subset of prompts
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        my_prompt_indices = list(range(rank, len(prompts), world_size))
        my_prompts = [prompts[i] for i in my_prompt_indices]

        # Cache tokenizer on first call (FSDP workers persist across steps)
        if not hasattr(self, "_hf_cached_tokenizer"):
            model_path = self.config.model.path
            self._hf_cached_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        tokenizer = self._hf_cached_tokenizer
        actor_module = self.actor.engine.module
        if isinstance(actor_module, torch.nn.parallel.DistributedDataParallel):
            actor_module = actor_module.module
        info_file = meta_info["info_file"]
        beam_width = int(meta_info.get("beam_width", 16))
        do_sample = bool(meta_info.get("do_sample", True))
        temperature = float(meta_info.get("temperature", 1.0))
        max_new_tokens = int(meta_info.get("max_new_tokens", 128))
        is_validate = bool(meta_info.get("validate", False))

        gen = HfConstrainedBeamGenerator(
            info_file=info_file,
            tokenizer=tokenizer,
            beam_width=beam_width,
            val_beam_width=int(meta_info.get("val_beam_width", beam_width)),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            micro_batch_size=int(meta_info.get("hf_micro_batch_size", 16)),
        )

        prompt_token_ids = meta_info.get("prompt_token_ids")
        if prompt_token_ids is not None:
            my_prompt_ids = [prompt_token_ids[i] for i in my_prompt_indices]

        actor_module.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(actor_module, FSDP):
            param_ctx = FSDP.summon_full_params(actor_module, writeback=False, recurse=False)

        with param_ctx, torch.inference_mode():
            if do_sample and not is_validate:
                outputs = gen.generate_train(
                    actor_module, my_prompts, prompt_token_ids=my_prompt_ids if prompt_token_ids is not None else None
                )
            else:
                outputs = gen.generate_eval(
                    actor_module, my_prompts, prompt_token_ids=my_prompt_ids if prompt_token_ids is not None else None
                )

        actor_module.train()

        grouped_resp_ids = [out.response_token_ids for out in outputs]
        grouped_decoded = [out.decoded_completions for out in outputs]

        return {
            "prompt_indices": my_prompt_indices,
            "response_ids": grouped_resp_ids,
            "decoded_completions": grouped_decoded,
        }
