"""Apply MiniOneRec training patches to verl FSDP engines (completion-only logprob + bnb optim)."""

from __future__ import annotations

_PATCHED = False


def apply_minionerec_engine_patches() -> None:
    """Patch ``FSDPEngineWithLMHead`` / ``FSDPEngine._build_optimizer`` once per process."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import verl.workers.engine.fsdp.transformer_impl as fsdp_impl
    from verl_gr.workers.engine.completion_only_logprob import CompletionOnlyLogprobMixin
    from verl_gr.workers.optimizer import build_actor_optimizer

    if not issubclass(fsdp_impl.FSDPEngineWithLMHead, CompletionOnlyLogprobMixin):

        class MiniOneRecFSDPEngineWithLMHead(CompletionOnlyLogprobMixin, fsdp_impl.FSDPEngineWithLMHead):
            pass

        fsdp_impl.FSDPEngineWithLMHead = MiniOneRecFSDPEngineWithLMHead

    if not getattr(fsdp_impl.FSDPEngine._build_optimizer, "_minionerec_optimizer_patch", False):

        def _build_optimizer(self, module):
            return build_actor_optimizer(module.parameters(), self.optimizer_config)

        _build_optimizer._minionerec_optimizer_patch = True
        fsdp_impl.FSDPEngine._build_optimizer = _build_optimizer
