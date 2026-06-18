"""Reusable async beam-search backend helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BeamCandidate:
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    cumulative_logprob: float = 0.0
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    def extend(self, token_id: int, token_logprob: float) -> "BeamCandidate":
        return BeamCandidate(
            prompt_token_ids=self.prompt_token_ids,
            generated_token_ids=self.generated_token_ids + [token_id],
            log_probs=self.log_probs + [token_logprob],
            cumulative_logprob=self.cumulative_logprob + token_logprob,
            finish_reason=self.finish_reason,
            stop_reason=self.stop_reason,
        )

    @property
    def full_prompt_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.generated_token_ids


def beam_search_score(
    candidate: BeamCandidate,
    *,
    eos_token_id: int,
    length_penalty: float,
) -> float:
    seq_len = len(candidate.full_prompt_token_ids)
    if candidate.generated_token_ids and candidate.generated_token_ids[-1] == eos_token_id:
        seq_len -= 1
    seq_len = max(seq_len, 1)
    return candidate.cumulative_logprob / (seq_len**length_penalty)


async def run_async_beam_search(
    *,
    prompt_token_ids: list[int],
    beam_width: int,
    max_tokens: int,
    eos_token_id: int,
    ignore_eos: bool,
    length_penalty: float,
    temperature: float = 1.0,
    generate_one_token: Callable[[list[int], str], Awaitable[Any]] | None = None,
    generate_next_tokens: Callable[[list[list[int]], list[str], list[list[int]] | None], Awaitable[list[Any]]] | None = None,
    allowed_tokens_fn: Callable[[list[int], list[int]], list[int]] | None = None,
    decode_mode: str = "deterministic_beam",
) -> list[BeamCandidate]:
    if generate_next_tokens is None:
        if generate_one_token is None:
            raise ValueError("Either generate_one_token or generate_next_tokens must be provided.")

        async def generate_next_tokens(
            prompt_token_ids_list: list[list[int]],
            request_suffixes: list[str],
            allowed_token_ids_list: list[list[int]] | None = None,  # noqa: ARG001
        ) -> list[Any]:
            tasks = [
                asyncio.create_task(generate_one_token(prompt_token_ids, request_suffix))
                for prompt_token_ids, request_suffix in zip(prompt_token_ids_list, request_suffixes, strict=True)
            ]
            return await asyncio.gather(*tasks)

    if decode_mode == "stochastic_constrained":
        active = [BeamCandidate(prompt_token_ids=list(prompt_token_ids)) for _ in range(max(1, beam_width))]
    else:
        active = [BeamCandidate(prompt_token_ids=list(prompt_token_ids))]
    completed: list[BeamCandidate] = []
    logprobs_num = max(2 * beam_width, 1)

    def add_fallback_token(beam: BeamCandidate, allowed_tokens: set[int], expanded: list[BeamCandidate]) -> bool:
        """Force a legal token when vLLM top-logprobs miss constrained tokens."""

        logger.warning(
            "add_fallback_token triggered: vLLM allowed_token_ids may not be fully enforced. "
            "beam_step=%d, allowed_count=%d",
            len(beam.generated_token_ids), len(allowed_tokens) if allowed_tokens else 0,
        )
        if allowed_tokens:
            token_id = eos_token_id if eos_token_id in allowed_tokens else min(allowed_tokens)
        else:
            token_id = eos_token_id
        if token_id is None:
            return False
        next_beam = beam.extend(int(token_id), 0.0)
        if token_id == eos_token_id and not ignore_eos:
            next_beam.finish_reason = "stop"
            next_beam.stop_reason = eos_token_id
            completed.append(next_beam)
        else:
            expanded.append(next_beam)
        return True

    for step in range(max_tokens):
        prompt_token_ids_list = [beam.full_prompt_token_ids for beam in active]
        request_suffixes = [f"beam-step-{step}-{beam_idx}-{uuid4().hex}" for beam_idx, _ in enumerate(active)]
        allowed_token_ids_list = None
        if allowed_tokens_fn is not None:
            allowed_token_ids_list = []
            for beam in active:
                allowed_token_ids = list(allowed_tokens_fn(beam.prompt_token_ids, beam.generated_token_ids))
                if not allowed_token_ids and eos_token_id is not None:
                    allowed_token_ids = [int(eos_token_id)]
                allowed_token_ids_list.append(allowed_token_ids)

        outputs = await generate_next_tokens(prompt_token_ids_list, request_suffixes, allowed_token_ids_list)

        expanded: list[BeamCandidate] = []
        for beam_idx, (beam, output) in enumerate(zip(active, outputs, strict=True)):
            allowed_tokens = set(allowed_token_ids_list[beam_idx]) if allowed_token_ids_list is not None else None
            if not output.outputs:
                if allowed_tokens is not None:
                    add_fallback_token(beam, allowed_tokens, expanded)
                continue
            first_output = output.outputs[0]
            if first_output.finish_reason == "error":
                raise RuntimeError("Async beam search received an error finish_reason from vLLM.")
            if not first_output.logprobs:
                if first_output.token_ids:
                    token_id = int(first_output.token_ids[0])
                    token_logprob = 0.0
                    next_beam = beam.extend(token_id, token_logprob)
                    expanded.append(next_beam)
                elif allowed_tokens is not None:
                    add_fallback_token(beam, allowed_tokens, expanded)
                continue

            step_logprobs = first_output.logprobs[0]
            if decode_mode == "stochastic_constrained" and first_output.token_ids:
                sampled_token = int(first_output.token_ids[0])
                sampled_info = step_logprobs.get(sampled_token)
                if sampled_info is None:
                    if allowed_tokens is not None:
                        add_fallback_token(beam, allowed_tokens, expanded)
                    continue
                ranked_tokens = [(sampled_token, sampled_info)]
            else:
                ranked_tokens = sorted(
                    step_logprobs.items(),
                    key=lambda item: item[1].logprob,
                    reverse=True,
                )[:logprobs_num]
            if allowed_tokens is not None:
                ranked_tokens = [(token_id, token_info) for token_id, token_info in ranked_tokens if int(token_id) in allowed_tokens]
                if not ranked_tokens:
                    add_fallback_token(beam, allowed_tokens, expanded)
                    continue

            for token_id, token_info in ranked_tokens:
                next_beam = beam.extend(int(token_id), float(token_info.logprob))
                if token_id == eos_token_id and not ignore_eos:
                    next_beam.finish_reason = "stop"
                    next_beam.stop_reason = eos_token_id
                    completed.append(next_beam)
                else:
                    expanded.append(next_beam)

        if not expanded:
            break

        expanded.sort(
            key=lambda candidate: beam_search_score(
                candidate,
                eos_token_id=eos_token_id,
                length_penalty=length_penalty,
            ),
            reverse=True,
        )
        active = expanded[:beam_width]

    # Fill with active (pre-EOS) beams only when EOS completions did not cover
    # the requested width.  Dropping them collapses the effective beam width,
    # while adding them after the width is already full only creates duplicates.
    remaining_slots = max(0, beam_width - len(completed))
    for beam in active[:remaining_slots]:
        if beam.finish_reason is None:
            beam.finish_reason = "length"
        completed.append(beam)

    if not completed and allowed_tokens_fn is not None:
        add_fallback_token(active[0], set(allowed_tokens_fn(active[0].prompt_token_ids, active[0].generated_token_ids)), completed)

    completed.sort(
        key=lambda candidate: beam_search_score(
            candidate,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
        ),
        reverse=True,
    )
    return completed[:beam_width]
