import math
import asyncio
from types import SimpleNamespace

import numpy as np

from verl_gr.workers.rollout.beam_backend import run_async_beam_search
from verl_gr.recipes.minionerec.minionerec_format import (
    build_seq_title2sid_prompt,
    build_sid_prompt,
    build_title2sid_prompt,
    parse_maybe_list,
)
from verl_gr.recipes.minionerec.minionerec_reward import compute_score, ndcg_penalties, normalize_sid


def test_sid_dataset_prompt_matches_minionerec_template():
    prompt, history_key = build_sid_prompt(["<a_1><b_2><c_3>", "<a_4><b_5><c_6>"])
    assert history_key == "<a_1><b_2><c_3>::<a_4><b_5><c_6>"
    assert prompt == (
        "### User Input: \n"
        "The user has interacted with items <a_1><b_2><c_3>, <a_4><b_5><c_6> in chronological order. "
        "Can you predict the next possible item that the user may expect?\n\n"
        "### Response:\n"
    )


def test_parse_maybe_list_accepts_minionerec_csv_lists():
    assert parse_maybe_list("['<a_1><b_2><c_3>']") == ["<a_1><b_2><c_3>"]
    assert parse_maybe_list(np.array(["x", "y"])) == ["x", "y"]


def test_title2sid_prompt_matches_minionerec_template():
    prompt, history_key = build_title2sid_prompt("title2sid", "A useful tool")
    assert history_key == "A useful tool"
    assert prompt == "### User Input: \nWhich item has the title: A useful tool?\n\n### Response:\n"

    prompt, history_key = build_title2sid_prompt("description2sid", "A small part")
    assert history_key == "A small part"
    assert prompt == '### User Input: \nAn item can be described as follows: "A small part". Which item is it describing?\n\n### Response:\n'


def test_seq_title2sid_prompt_matches_minionerec_template():
    prompt, history_key = build_seq_title2sid_prompt(["Hammer", "Screwdriver"])
    assert history_key == "Hammer::Screwdriver"
    assert prompt == (
        '### User Input: \nGiven the title sequence of user historical interactive items: "Hammer", "Screwdriver", '
        "can you recommend a suitable next item for the user?\n\n### Response:\n"
    )


def test_sample_level_reward_normalization():
    assert normalize_sid('" <a_1><b_2><c_3>\n" ') == "<a_1><b_2><c_3>"
    score = compute_score("minionerec", "<a_1><b_2><c_3>\n", "<a_1><b_2><c_3>\n", {})
    assert score["score"] == 1.0
    assert score["valid_sid"] == 1.0


def test_ndcg_penalties_match_minionerec_formula():
    penalties = ndcg_penalties(4)
    raw = [-1.0 / math.log2(i + 2) for i in range(4)]
    expected = [(-value / sum(raw)) for value in raw]
    assert penalties == expected
    assert all(value < 0 for value in penalties)


def test_constrained_beam_falls_back_to_eos_when_top_logprobs_miss_allowed_tokens():
    async def generate_one_token(_prompt_ids, _request_id):
        token_info = SimpleNamespace(logprob=-1.0)
        output = SimpleNamespace(outputs=[SimpleNamespace(finish_reason=None, logprobs=[{7: token_info}], token_ids=[7])])
        return output

    beams = asyncio.run(
        run_async_beam_search(
            prompt_token_ids=[1, 2, 3],
            beam_width=1,
            max_tokens=4,
            eos_token_id=9,
            ignore_eos=False,
            length_penalty=1.0,
            generate_one_token=generate_one_token,
            allowed_tokens_fn=lambda _prompt, _generated: [9],
        )
    )

    assert beams[0].generated_token_ids == [9]
    assert beams[0].finish_reason == "stop"


def test_constrained_beam_falls_back_to_allowed_token_when_top_logprobs_miss_non_eos():
    async def generate_one_token(_prompt_ids, _request_id):
        token_info = SimpleNamespace(logprob=-1.0)
        output = SimpleNamespace(outputs=[SimpleNamespace(finish_reason=None, logprobs=[{7: token_info}], token_ids=[7])])
        return output

    beams = asyncio.run(
        run_async_beam_search(
            prompt_token_ids=[1, 2, 3],
            beam_width=1,
            max_tokens=1,
            eos_token_id=9,
            ignore_eos=False,
            length_penalty=1.0,
            generate_one_token=generate_one_token,
            allowed_tokens_fn=lambda _prompt, _generated: [8],
        )
    )

    assert beams[0].generated_token_ids == [8]


def test_batched_beam_passes_allowed_token_ids_to_batch_api():
    captured_allowed = []

    async def generate_next_tokens(_prompt_ids_list, _request_suffixes, allowed_token_ids_list):
        captured_allowed.extend(allowed_token_ids_list)
        token_info = SimpleNamespace(logprob=-1.0)
        return [
            SimpleNamespace(outputs=[SimpleNamespace(finish_reason=None, logprobs=[{8: token_info}], token_ids=[8])])
        ]

    beams = asyncio.run(
        run_async_beam_search(
            prompt_token_ids=[1, 2, 3],
            beam_width=1,
            max_tokens=1,
            eos_token_id=9,
            ignore_eos=False,
            length_penalty=1.0,
            generate_next_tokens=generate_next_tokens,
            allowed_tokens_fn=lambda _prompt, _generated: [8],
        )
    )

    assert captured_allowed == [[8]]
    assert beams[0].generated_token_ids == [8]


def test_unconstrained_batched_beam_does_not_pass_allowed_token_ids():
    captured_allowed = ["not-called"]

    async def generate_next_tokens(_prompt_ids_list, _request_suffixes, allowed_token_ids_list):
        captured_allowed.clear()
        captured_allowed.append(allowed_token_ids_list)
        token_info = SimpleNamespace(logprob=-1.0)
        return [
            SimpleNamespace(outputs=[SimpleNamespace(finish_reason=None, logprobs=[{7: token_info}], token_ids=[7])])
        ]

    beams = asyncio.run(
        run_async_beam_search(
            prompt_token_ids=[1, 2, 3],
            beam_width=1,
            max_tokens=1,
            eos_token_id=9,
            ignore_eos=False,
            length_penalty=1.0,
            generate_next_tokens=generate_next_tokens,
        )
    )

    assert captured_allowed == [None]
    assert beams[0].generated_token_ids == [7]


def test_stochastic_constrained_mode_uses_sampled_token_not_top_logprob():
    async def generate_next_tokens(_prompt_ids_list, _request_suffixes, _allowed_token_ids_list):
        top_token = SimpleNamespace(logprob=-0.01)
        sampled_token = SimpleNamespace(logprob=-3.0)
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        finish_reason=None,
                        logprobs=[{7: top_token, 8: sampled_token}],
                        token_ids=[8],
                    )
                ]
            )
            for _ in _prompt_ids_list
        ]

    beams = asyncio.run(
        run_async_beam_search(
            prompt_token_ids=[1, 2],
            beam_width=2,
            max_tokens=1,
            eos_token_id=9,
            ignore_eos=False,
            length_penalty=1.0,
            generate_next_tokens=generate_next_tokens,
            decode_mode="stochastic_constrained",
        )
    )

    assert len(beams) == 2
    assert all(beam.generated_token_ids == [8] for beam in beams)


def test_beam_search_keeps_active_beams_when_some_complete_early():
    async def generate_next_tokens(prompt_ids_list, _request_suffixes, _allowed_token_ids_list):
        results = []
        for prompt_ids in prompt_ids_list:
            if prompt_ids[-1] == 7:
                results.append(
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                finish_reason=None,
                                logprobs=[{8: SimpleNamespace(logprob=-0.3)}],
                                token_ids=[8],
                            )
                        ]
                    )
                )
            else:
                results.append(
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                finish_reason=None,
                                logprobs=[
                                    {
                                        9: SimpleNamespace(logprob=-0.1),
                                        7: SimpleNamespace(logprob=-0.2),
                                    }
                                ],
                                token_ids=[9],
                            )
                        ]
                    )
                )
        return results

    beams = asyncio.run(
        run_async_beam_search(
            prompt_token_ids=[1, 2],
            beam_width=2,
            max_tokens=2,
            eos_token_id=9,
            ignore_eos=False,
            length_penalty=0.0,
            generate_next_tokens=generate_next_tokens,
        )
    )

    assert len(beams) == 2
    assert [beam.generated_token_ids for beam in beams] == [[9], [7, 8]]
    assert [beam.finish_reason for beam in beams] == ["stop", "length"]
