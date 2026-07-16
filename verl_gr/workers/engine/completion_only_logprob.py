"""Completion-only logprob path aligned with MiniOneRec ``logits_to_keep``.

Primary fast-path is rmpad (nested inputs): run normal rmpad forward, then
compute logprobs only on completion token positions via gathered rows from
``logits_rmpad``. This avoids full-token LM-head logprob work in actor update.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import logprobs_from_logits

_LOGPROB_DUMP_COUNTS: dict[int, int] = {}


@contextmanager
def _nvtx_range(name: str):
    enabled = torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log p(label | logits), matching TRL ``selective_log_softmax``."""
    log_probs = logits.log_softmax(dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def per_token_logps_logits_to_keep(
    logits: torch.Tensor, input_ids: torch.Tensor, logits_to_keep: int
) -> torch.Tensor:
    """Mirror ``MiniOneRec/minionerec_trainer._get_per_token_logps``."""
    logits = logits[:, :-1, :]
    completion_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, completion_ids)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _tensor_summary(tensor: torch.Tensor | None) -> dict | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    t = tensor.detach()
    out = {"shape": list(t.shape), "dtype": str(t.dtype)}
    if t.numel() == 0:
        return out
    if t.is_floating_point():
        f = t.float()
        out.update(
            mean=float(f.mean().item()),
            std=float(f.std(unbiased=False).item()) if f.numel() > 1 else 0.0,
            min=float(f.min().item()),
            max=float(f.max().item()),
        )
    else:
        out.update(min=int(t.min().item()), max=int(t.max().item()), checksum=int(t.long().sum().item()))
    return out


def _max_response_len(micro_batch: TensorDict) -> int:
    max_response_len = tu.get_non_tensor_data(micro_batch, key="max_response_len", default=-1)
    if max_response_len is not None and int(max_response_len) > 0:
        return int(max_response_len)
    loss_mask = micro_batch.get("loss_mask")
    if loss_mask is not None:
        if loss_mask.is_nested:
            return int(loss_mask.offsets().diff().max().item())
        # tensor loss_mask is padded; use true per-sample valid length.
        return int(loss_mask.sum(dim=1).max().item())
    responses = micro_batch.get("responses")
    if responses is not None:
        if responses.is_nested:
            return int(responses.offsets().diff().max().item())
        pad_token_id = int(tu.get_non_tensor_data(micro_batch, key="pad_token_id", default=0) or 0)
        return int((responses != pad_token_id).sum(dim=1).max().item())
    return 16


def _prompt_and_response_lens(micro_batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve per-sample prompt/response lengths after ``left_right_2_no_padding``.

    ``input_ids`` is nested jagged; ``prompts``/``responses`` stay left-right padded tensors.
    """
    input_ids = micro_batch["input_ids"]
    if not input_ids.is_nested:
        raise TypeError(f"completion_only_logprob expects nested input_ids, got {type(input_ids)}")

    seq_lens = input_ids.offsets().diff().to(torch.long)
    loss_mask = micro_batch.get("loss_mask")
    responses = micro_batch.get("responses")
    prompts = micro_batch.get("prompts")

    if responses is not None and responses.is_nested:
        response_lens = responses.offsets().diff().to(torch.long)
        if prompts is not None and prompts.is_nested:
            prompt_lens = prompts.offsets().diff().to(torch.long)
        else:
            prompt_lens = seq_lens - response_lens
        return prompt_lens, response_lens

    if loss_mask is not None:
        if loss_mask.is_nested:
            response_lens = loss_mask.offsets().diff().to(torch.long)
        else:
            response_lens = loss_mask.sum(dim=1).to(torch.long).clamp(min=0)
        prompt_lens = seq_lens - response_lens
        return prompt_lens, response_lens

    if responses is not None:
        pad_token_id = int(tu.get_non_tensor_data(micro_batch, key="pad_token_id", default=0) or 0)
        response_lens = (responses != pad_token_id).sum(dim=1).to(torch.long).clamp(min=0)
        if prompts is not None:
            prompt_lens = (prompts != pad_token_id).sum(dim=1).to(torch.long).clamp(min=0)
        else:
            prompt_lens = seq_lens - response_lens
        return prompt_lens, response_lens

    raise KeyError(
        "completion_only_logprob needs nested input_ids plus loss_mask (response_mask) "
        "or prompts/responses tensors in the micro-batch"
    )


def nested_log_probs_from_completion_logps(
    completion_logps: torch.Tensor,
    micro_batch: TensorDict,
) -> torch.Tensor:
    """Pack (bsz, max_resp) completion logps into nested (bsz, seqlen) for ``no_padding_2_padding``."""
    input_ids = micro_batch["input_ids"]
    assert input_ids.is_nested

    prompt_lens, response_lens = _prompt_and_response_lens(micro_batch)
    cu_seqlens = input_ids.offsets()
    total_nnz = int(cu_seqlens[-1].item())
    flat = completion_logps.new_zeros((total_nnz,))

    offset = 0
    for i, (p_len, r_len) in enumerate(zip(prompt_lens, response_lens, strict=True)):
        p_len = int(p_len.item())
        r_len = int(r_len.item())
        if r_len > 0 and p_len > 0:
            n = min(r_len, completion_logps.shape[1])
            flat[offset + p_len - 1 : offset + p_len - 1 + n] = completion_logps[i, :n]
        offset += p_len + r_len

    return torch.nested.nested_tensor_from_jagged(flat, offsets=cu_seqlens)


def _build_completion_logit_indices(micro_batch: TensorDict) -> torch.Tensor:
    """Build flattened rmpad indices for response-token logprobs.

    For each sample with prompt length ``p`` and response length ``r``, completion
    logprobs are at flattened positions ``[offset + p - 1, ..., offset + p + r - 2]``.
    """
    input_ids = micro_batch["input_ids"]
    assert input_ids.is_nested
    prompt_lens, response_lens = _prompt_and_response_lens(micro_batch)
    cu_seqlens = input_ids.offsets().to(torch.long)
    idx_chunks: list[torch.Tensor] = []
    for i, (p_len, r_len) in enumerate(zip(prompt_lens, response_lens, strict=True)):
        p = int(p_len.item())
        r = int(r_len.item())
        if p <= 0 or r <= 0:
            continue
        start = int(cu_seqlens[i].item()) + p - 1
        idx_chunks.append(torch.arange(start, start + r, dtype=torch.long, device=cu_seqlens.device))
    if not idx_chunks:
        return torch.empty((0,), dtype=torch.long, device=cu_seqlens.device)
    return torch.cat(idx_chunks, dim=0)


def _scatter_selected_log_probs(selected_log_probs: torch.Tensor, indices: torch.Tensor, micro_batch: TensorDict) -> torch.Tensor:
    """Scatter selected completion logprobs back to nested full-length tensor."""
    input_ids = micro_batch["input_ids"]
    cu_seqlens = input_ids.offsets()
    total_nnz = int(cu_seqlens[-1].item())
    flat = selected_log_probs.new_zeros((total_nnz,))
    if indices.numel() > 0:
        flat.index_copy_(0, indices, selected_log_probs)
    return torch.nested.nested_tensor_from_jagged(flat, offsets=cu_seqlens)


def _build_padded_completion_inputs(micro_batch: TensorDict, max_response_len: int):
    """Build padded inputs for forward-only ref path with logits_to_keep."""
    input_ids = micro_batch["input_ids"]
    position_ids = micro_batch["position_ids"]
    pad_token_id = int(tu.get_non_tensor_data(micro_batch, key="pad_token_id", default=0) or 0)
    batch_size = micro_batch.batch_size[0]
    logits_to_keep = max_response_len + 1

    # If the original padded tensors are still present, prefer them.  This
    # mirrors MiniOneRec ReReTrainer's left-padded forward exactly:
    # input_ids = cat(prompt_ids, completion_ids), attention_mask = cat(...).
    prompts = micro_batch.get("prompts")
    responses = micro_batch.get("responses")
    original_attention_mask = micro_batch.get("attention_mask")
    if (
        isinstance(prompts, torch.Tensor)
        and isinstance(responses, torch.Tensor)
        and not prompts.is_nested
        and not responses.is_nested
    ):
        # Dataset / rollout often left-pads prompts to data.max_prompt_length
        # (e.g. 2560). Original MiniOneRec collates with padding=True so the
        # forward sees only batch-max content length. Trim leading pads here
        # before cat; otherwise force_padded SDPA OOMs on the full pad window.
        prompt_mask = None
        if (
            isinstance(original_attention_mask, torch.Tensor)
            and not original_attention_mask.is_nested
            and original_attention_mask.shape[-1] >= prompts.shape[-1]
        ):
            prompt_mask = original_attention_mask[:, : prompts.shape[-1]].to(torch.int32)
        else:
            prompt_mask = (prompts != pad_token_id).to(torch.int32)
        content_lens = prompt_mask.sum(dim=-1)
        keep_prompt = int(content_lens.max().item()) if content_lens.numel() else 0
        keep_prompt = max(keep_prompt, 1)
        if keep_prompt < prompts.shape[-1]:
            prompts = prompts[:, -keep_prompt:].contiguous()
            prompt_mask = prompt_mask[:, -keep_prompt:].contiguous()
        response_mask = (responses != pad_token_id).to(torch.int32)
        if (
            isinstance(original_attention_mask, torch.Tensor)
            and not original_attention_mask.is_nested
            and original_attention_mask.shape[-1] >= prompts.shape[-1] + responses.shape[-1]
        ):
            # Prefer response slice from the original full mask when available.
            response_mask = original_attention_mask[:, -responses.shape[-1] :].to(torch.int32)
        input_ids_padded = torch.cat([prompts, responses], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        position_ids_padded = None
    else:
        max_seq_len = int(input_ids.offsets().diff().max().item())
        input_ids_padded = torch.nested.to_padded_tensor(
            input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
        )
        if position_ids.dim() == 3:
            position_ids_padded = torch.nested.to_padded_tensor(
                position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
            ).transpose(0, 1)
        else:
            position_ids_padded = torch.nested.to_padded_tensor(
                position_ids, padding=0, output_size=(batch_size, max_seq_len)
            )
        attention_mask = (input_ids_padded != pad_token_id).to(torch.int32)

    temperature = micro_batch["temperature"]
    if not isinstance(temperature, torch.Tensor):
        temperature = torch.tensor([temperature] * batch_size, device=input_ids_padded.device)
    temperature = temperature.to(torch.float32)
    model_inputs = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "logits_to_keep": logits_to_keep,
    }
    if position_ids_padded is not None:
        model_inputs["position_ids"] = position_ids_padded
    output_args = {
        "completion_only": True,
        "completion_only_padded": True,
        "logits_to_keep": logits_to_keep,
        "input_ids_padded": input_ids_padded,
        "attention_mask_padded": attention_mask,
        "temperature": temperature,
    }
    return model_inputs, output_args


class CompletionOnlyLogprobMixin:
    """Override LM-head logprob to use ``logits_to_keep`` (completion tokens only)."""

    def _logprob_param_summaries(self) -> dict[str, dict]:
        module = getattr(self, "module", None)
        if module is None:
            return {}
        if hasattr(module, "module"):
            module = module.module
        substrings = os.getenv(
            "MINIONEREC_LOGPROB_DUMP_PARAM_SUBSTR",
            "model.embed_tokens.weight,model.layers.0.self_attn.q_proj.weight,model.layers.27.mlp.down_proj.weight,lm_head.weight",
        ).split(",")
        substrings = [item.strip() for item in substrings if item.strip()]
        out = {}
        for name, param in module.named_parameters():
            if any(substr in name for substr in substrings):
                data = param.detach().float()
                flat = data.reshape(-1)
                sample = flat[: min(16, flat.numel())].cpu()
                out[name] = {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "norm": float(data.norm().item()),
                    "mean": float(data.mean().item()),
                    "sample": sample.tolist(),
                }
        return out

    def _maybe_dump_logprob_microbatch(
        self,
        *,
        backend: str,
        logits: torch.Tensor,
        input_ids_padded: torch.Tensor,
        attention_mask: torch.Tensor | None,
        completion_logps: torch.Tensor,
        logits_to_keep: int,
    ) -> None:
        dump_dir = os.getenv("MINIONEREC_LOGPROB_DUMP_DIR")
        if not dump_dir:
            return
        rank = _rank()
        count = _LOGPROB_DUMP_COUNTS.get(rank, 0)
        max_dumps = int(os.getenv("MINIONEREC_LOGPROB_DUMP_MAX", "1"))
        if count >= max_dumps:
            return
        _LOGPROB_DUMP_COUNTS[rank] = count + 1
        rows = min(int(os.getenv("MINIONEREC_LOGPROB_DUMP_ROWS", "8")), int(input_ids_padded.shape[0]))
        topk = int(os.getenv("MINIONEREC_LOGPROB_DUMP_TOPK", "5"))
        os.makedirs(dump_dir, exist_ok=True)
        role = "ref" if bool(getattr(self.engine_config, "forward_only", False)) else "actor"
        pid = os.getpid()

        completion_logits = logits[:, :-1, :][:, -(logits_to_keep - 1) :, :]
        labels = input_ids_padded[:, -(logits_to_keep - 1) :]
        sample_logits = completion_logits[:rows].detach().float()
        sample_labels = labels[:rows].to(sample_logits.device)
        label_logits = sample_logits.gather(-1, sample_labels.unsqueeze(-1)).squeeze(-1)
        logsumexp = torch.logsumexp(sample_logits, dim=-1)
        topk_values, topk_indices = torch.topk(sample_logits, k=min(topk, sample_logits.shape[-1]), dim=-1)

        payload = {
            "backend": backend,
            "role": role,
            "pid": pid,
            "rank": rank,
            "count": count,
            "logits_to_keep": int(logits_to_keep),
            "input_ids_padded": input_ids_padded[:rows].detach().cpu().clone(),
            "attention_mask": None if attention_mask is None else attention_mask[:rows].detach().cpu().clone(),
            "labels": labels[:rows].detach().cpu().clone(),
            "label_logits": label_logits.detach().cpu(),
            "logsumexp": logsumexp.detach().cpu(),
            "completion_logps": completion_logps[:rows].detach().float().cpu(),
            "topk_values": topk_values.detach().cpu(),
            "topk_indices": topk_indices.detach().cpu(),
            "logits_summary": _tensor_summary(sample_logits),
            "param_summaries": self._logprob_param_summaries(),
        }
        path = os.path.join(dump_dir, f"logprob_{role}_{backend}_rank{rank}_pid{pid}_dump{count}.pt")
        torch.save(payload, path)
        summary = {
            key: value
            for key, value in payload.items()
            if key not in {"input_ids_padded", "attention_mask", "labels", "label_logits", "logsumexp", "completion_logps", "topk_values", "topk_indices"}
        }
        summary["path"] = path
        with open(path.replace(".pt", ".json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=str)

    def _completion_only_logprob_enabled(self, micro_batch: TensorDict) -> bool:
        if getattr(self.engine_config, "completion_only_logprob", False):
            return True
        return bool(tu.get_non_tensor_data(micro_batch, key="completion_only_logprob", default=False))

    def prepare_model_inputs(self, micro_batch: TensorDict):
        if not self._completion_only_logprob_enabled(micro_batch):
            return super().prepare_model_inputs(micro_batch)

        model_inputs, output_args = super().prepare_model_inputs(micro_batch)

        pad_mode = tu.get_non_tensor_data(micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_remove_padding = tu.get_non_tensor_data(micro_batch, key="use_remove_padding", default=True)
        use_fused_kernels = tu.get_non_tensor_data(micro_batch, key="use_fused_kernels", default=False)
        max_response_len = _max_response_len(micro_batch)
        force_padded = bool(getattr(self.engine_config, "completion_only_force_padded", False))

        # Original MiniOneRec uses padded HF forward with logits_to_keep.  When
        # remove-padding is disabled, keep that contract instead of falling back
        # to full-sequence logits.
        if pad_mode != DatasetPadMode.NO_PADDING or not micro_batch["input_ids"].is_nested:
            logits_to_keep = max_response_len + 1
            model_inputs["logits_to_keep"] = logits_to_keep
            if bool(getattr(self.engine_config, "completion_only_drop_position_ids", False)):
                model_inputs.pop("position_ids", None)
            temperature = micro_batch.get("temperature", 1.0)
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.tensor(
                    [float(temperature)] * micro_batch.batch_size[0],
                    device=micro_batch["input_ids"].device,
                )
            output_args["completion_only"] = True
            output_args["completion_only_padded"] = True
            output_args["logits_to_keep"] = logits_to_keep
            output_args["input_ids_padded"] = micro_batch["input_ids"]
            output_args["attention_mask_padded"] = model_inputs.get("attention_mask")
            output_args["temperature"] = temperature.to(torch.float32)
            return model_inputs, output_args

        # MiniOneRec parity path: use padded + logits_to_keep only when the
        # launcher explicitly requests it. Forward-only ref logprob can otherwise
        # use the same rmpad completion-only fast path as the actor.
        if force_padded and pad_mode == DatasetPadMode.NO_PADDING and micro_batch["input_ids"].is_nested:
            model_inputs, output_args = _build_padded_completion_inputs(micro_batch, max_response_len)
            if bool(getattr(self.engine_config, "completion_only_drop_position_ids", False)):
                model_inputs.pop("position_ids", None)
            return model_inputs, output_args

        # Fast path: rmpad + non-fused kernels. Keep original rmpad inputs and only
        # compute completion-token logprobs from gathered logits rows.
        if (
            pad_mode == DatasetPadMode.NO_PADDING
            and use_remove_padding
            and not use_fused_kernels
            and "input_ids_rmpad_rolled" in output_args
            and micro_batch["input_ids"].is_nested
        ):
            completion_indices = _build_completion_logit_indices(micro_batch).to(
                output_args["input_ids_rmpad_rolled"].device
            )
            output_args["completion_only"] = True
            output_args["completion_only_rmpad"] = True
            output_args["completion_logit_indices"] = completion_indices
            output_args["max_response_len"] = max_response_len
        else:
            # Fallback to default behavior for unsupported paths.
            output_args["completion_only"] = False
        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        if not output_args.get("completion_only", False):
            return super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)

        calculate_entropy = tu.get_non_tensor_data(micro_batch, key="calculate_entropy", default=False)
        if calculate_entropy:
            raise NotImplementedError("completion_only_logprob does not support calculate_entropy=True")

        if output_args.get("completion_only_rmpad", False):
            with _nvtx_range("logprob.completion_only"):
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab)
                temperature_rmpad = output_args["temperature_rmpad"]  # (total_nnz,)
                logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

                completion_indices = output_args["completion_logit_indices"]
                if completion_indices.numel() > 0:
                    selected_logits = logits_rmpad.index_select(0, completion_indices)
                    selected_labels = output_args["input_ids_rmpad_rolled"].index_select(0, completion_indices)
                    selected_log_probs = logprobs_from_logits(
                        logits=selected_logits,
                        labels=selected_labels,
                        inplace_backward=True,
                    )
                else:
                    selected_log_probs = logits_rmpad.new_empty((0,))

                log_probs = _scatter_selected_log_probs(selected_log_probs, completion_indices, micro_batch)
                return {"log_probs": log_probs}

        if output_args.get("completion_only_padded", False):
            with _nvtx_range("logprob.completion_only"):
                logits = output.logits
                logits_to_keep = int(output_args["logits_to_keep"])
                input_ids_padded = output_args["input_ids_padded"]
                temperature = output_args["temperature"]
                if logits.shape[1] > logits_to_keep:
                    logits = logits[:, -logits_to_keep:, :]
                logits = logits / temperature.clamp(min=1e-8).view(-1, 1, 1).to(logits.dtype)
                completion_logps = per_token_logps_logits_to_keep(logits, input_ids_padded, logits_to_keep - 1)
                self._maybe_dump_logprob_microbatch(
                    backend="padded",
                    logits=logits,
                    input_ids_padded=input_ids_padded,
                    attention_mask=output_args.get("attention_mask_padded"),
                    completion_logps=completion_logps,
                    logits_to_keep=logits_to_keep,
                )
                log_probs = nested_log_probs_from_completion_logps(completion_logps, micro_batch)
                return {"log_probs": log_probs}

        return super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)
