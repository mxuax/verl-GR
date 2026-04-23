"""Reusable async beam-search backend helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4


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
    generate_one_token: Callable[[list[int], str], Awaitable[Any]],
) -> list[BeamCandidate]:
    active = [BeamCandidate(prompt_token_ids=list(prompt_token_ids))]
    completed: list[BeamCandidate] = []
    logprobs_num = max(2 * beam_width, 1)

    for step in range(max_tokens):
        tasks = [
            asyncio.create_task(
                generate_one_token(
                    beam.full_prompt_token_ids,
                    f"beam-step-{step}-{beam_idx}-{uuid4().hex}",
                )
            )
            for beam_idx, beam in enumerate(active)
        ]
        outputs = await asyncio.gather(*tasks)

        expanded: list[BeamCandidate] = []
        for beam, output in zip(active, outputs, strict=True):
            if not output.outputs:
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
                continue

            step_logprobs = first_output.logprobs[0]
            ranked_tokens = sorted(
                step_logprobs.items(),
                key=lambda item: item[1].logprob,
                reverse=True,
            )[:logprobs_num]

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

    for beam in active:
        if beam.finish_reason is None:
            beam.finish_reason = "length"
        completed.append(beam)

    completed.sort(
        key=lambda candidate: beam_search_score(
            candidate,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
        ),
        reverse=True,
    )
    return completed[:beam_width]
