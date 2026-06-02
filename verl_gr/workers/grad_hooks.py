"""FSDP engine patches for verl-GR.

Always installed: fix for in-place div_ on FSDP2 logits (autograd disconnect).
Debug-only (VERL_GR_DEBUG=1): per-layer gradient norm logging to stderr and
tensorboard metrics.
"""

import os
import sys

import torch

_original_train_batch = None
_original_prepare_model_outputs = None


def _collect_layer_grad_norms(engine_self) -> dict[str, float]:
    """Iterate model parameters, group by layer, return L2 norm per layer."""
    layer_sq: dict[str, float] = {}
    total_sq = 0.0
    for name, param in engine_self.module.named_parameters():
        if param.grad is None:
            continue
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            layer_key = ".".join(parts[: idx + 2])
        else:
            layer_key = ".".join(parts[:3]) if len(parts) >= 3 else name
        g = param.grad.detach().float()
        sq = g.square().sum().item()
        layer_sq[layer_key] = layer_sq.get(layer_key, 0.0) + sq
        total_sq += sq

    try:
        dp_group = engine_self.get_data_parallel_group()
    except Exception:
        dp_group = None
    if dp_group is not None:
        keys = sorted(layer_sq.keys())
        device = next(engine_self.module.parameters()).device
        vals = torch.tensor(
            [layer_sq[k] for k in keys] + [total_sq], device=device
        )
        torch.distributed.all_reduce(vals, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        for i, k in enumerate(keys):
            layer_sq[k] = vals[i].item()
        total_sq = vals[-1].item()

    sorted_layers = sorted(layer_sq.items(), key=lambda x: -x[1])
    top5 = [(k, float(v ** 0.5)) for k, v in sorted_layers[:5]]
    bot5 = [(k, float(v ** 0.5)) for k, v in sorted_layers[-5:]]
    total_l2 = float(total_sq ** 0.5)
    print(
        f"[LAYER_GRAD] total_l2={total_l2:.6f} top5={top5} bot5={bot5}",
        file=sys.stderr,
        flush=True,
    )

    metrics = {"debug_grad/total_l2": [total_l2]}
    for k, v in layer_sq.items():
        if ".layers." in k:
            layer_num = k.split(".layers.")[-1]
            metrics[f"debug_grad/layer_{layer_num}"] = [float(v ** 0.5)]
        else:
            metrics[f"debug_grad/{k}"] = [float(v ** 0.5)]
    return metrics


def _patched_train_batch(self, data, loss_function):
    """Wrapper around FSDPEngine.train_batch that collects per-layer
    gradient norms after forward_backward_batch and before optimizer_step.
    """
    global _original_train_batch

    if _original_train_batch is None:
        raise RuntimeError("grad_hooks not installed — call install_grad_hooks() first")

    from verl.utils.tensordict_utils import maybe_fix_3d_position_ids

    maybe_fix_3d_position_ids(data)

    self.optimizer_zero_grad()

    _log_diag = os.environ.get("DEBUG_LAYER_GRAD", "0") == "1"
    if _log_diag:
        loss_mask = data.get("loss_mask", data.get("item_token_mask"))
        n_tokens = int(loss_mask.sum().item()) if loss_mask is not None else -1
        n_seqs = data.shape[0] if hasattr(data, 'shape') else -1
        adv = data.get("advantages")
        adv_hash = "none"
        if adv is not None:
            adv_flat = adv.detach().float()
            adv_hash = f"{adv_flat.mean().item():.6f}_{adv_flat.std().item():.6f}"

    outputs = self.forward_backward_batch(data, loss_function, forward_only=False)

    if _log_diag:
        layer_grads = _collect_layer_grad_norms(self)
        if layer_grads is not None:
            outputs["metrics"].update(layer_grads)
        loss_vals = [float(l.item()) if hasattr(l, 'item') else float(l) for l in outputs.get("loss", [])]
        grad_l2 = layer_grads.get("debug_grad/total_l2", [-1])[0] if layer_grads else -1
        n_micro = len(outputs.get("loss", []))
        print(
            f"[GRAD_LOSS] grad_l2={grad_l2:.6f} loss={sum(loss_vals):.6f} "
            f"n_tok={n_tokens} n_seq={n_seqs} n_micro={n_micro} adv_hash={adv_hash}",
            file=sys.stderr,
            flush=True,
        )

    grad_norm = self.optimizer_step()
    if self.is_mp_src_rank_with_outputs():
        assert "grad_norm" not in outputs["metrics"]
        outputs["metrics"]["grad_norm"] = grad_norm
    return outputs


def _patched_prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
    """Original prepare_model_outputs but with in-place div_ replaced by
    safe out-of-place division.  In-place arithmetic on FSDP2 output
    activations can invalidate autograd hooks, causing bimodal gradient
    collapse."""
    global _original_prepare_model_outputs

    _orig_div_ = torch.Tensor.div_

    def _safe_div_(tensor, other):
        result = tensor.div(other)
        return tensor.copy_(result)

    try:
        torch.Tensor.div_ = _safe_div_
        return _original_prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func)
    finally:
        torch.Tensor.div_ = _orig_div_


def install_grad_hooks():
    """Install FSDP engine patches.

    Always applied: fix for in-place div_ on FSDP2 logits (autograd fix).
    Debug-only (VERL_GR_DEBUG=1): per-layer gradient norm logging to stderr
    and tensorboard metrics.
    """
    global _original_train_batch, _original_prepare_model_outputs

    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead

    # --- Debug-only: gradient diagnostics and FSDP2 div_ autograd fix ---
    if os.environ.get("VERL_GR_DEBUG", "0") == "1":
        if _original_train_batch is None:
            os.environ["DEBUG_LAYER_GRAD"] = "1"
            _original_train_batch = FSDPEngine.train_batch
            FSDPEngine.train_batch = _patched_train_batch
            print("[grad_hooks] patched FSDPEngine.train_batch (VERL_GR_DEBUG=1)", flush=True)

        if _original_prepare_model_outputs is None:
            _original_prepare_model_outputs = FSDPEngineWithLMHead.prepare_model_outputs
            FSDPEngineWithLMHead.prepare_model_outputs = _patched_prepare_model_outputs
            print("[grad_hooks] patched FSDPEngineWithLMHead.prepare_model_outputs (fixed div_)", flush=True)
