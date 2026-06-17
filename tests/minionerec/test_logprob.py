"""Tests for completion-only logprob helpers."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from verl_gr.workers.engine.completion_only_logprob import (
    nested_log_probs_from_completion_logps,
    per_token_logps_logits_to_keep,
)


def test_per_token_logps_logits_to_keep_matches_slice_logic():
    bsz, resp_len, vocab = 2, 4, 32
    logits_to_keep = resp_len
    total_k = logits_to_keep + 1
    logits = torch.randn(bsz, total_k, vocab)
    input_ids = torch.randint(0, vocab, (bsz, 20))
    input_ids[:, -resp_len:] = torch.randint(0, vocab, (bsz, resp_len))

    logps = per_token_logps_logits_to_keep(logits, input_ids, logits_to_keep)
    assert logps.shape == (bsz, resp_len)


def test_nested_log_probs_layout_for_no_padding_extract():
    # Verl batch: nested input_ids + padded prompts/responses (post left_right_2_no_padding).
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([1, 2, 3, 7, 8]), torch.tensor([4, 5, 6, 9, 10])], layout=torch.jagged
    )
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    responses = torch.tensor([[7, 8, 0], [9, 10, 0]])
    loss_mask = torch.tensor([[1, 1], [1, 1]])
    completion_logps = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    micro_batch = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": prompts,
            "responses": responses,
            "loss_mask": loss_mask,
        },
        batch_size=2,
    )
    nested = nested_log_probs_from_completion_logps(completion_logps, micro_batch)
    flat = nested.values()
    assert flat.shape[0] == 10
    assert torch.allclose(flat[2:4], torch.tensor([0.1, 0.2]))
    assert torch.allclose(flat[7:9], torch.tensor([0.3, 0.4]))


if __name__ == "__main__":
    test_per_token_logps_logits_to_keep_matches_slice_logic()
    test_nested_log_probs_layout_for_no_padding_extract()
    print("test_minionerec_completion_only_logprob: ok")
