import importlib.util
import sys
import unittest
from pathlib import Path


def _load_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "eval" / "openonerec_eval.py"
    spec = importlib.util.spec_from_file_location("openonerec_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OpenOneRecEvalPromptTest(unittest.TestCase):
    def test_prepare_eval_prompt_messages_removes_labeled_answer(self):
        module = _load_eval_module()
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "recommend an item"},
            {"role": "assistant", "content": "<|sid_begin|>target<|sid_end|>"},
        ]

        prompt_messages, groundtruth = module._prepare_eval_prompt_messages(messages)

        self.assertEqual(
            prompt_messages,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "recommend an item"},
            ],
        )
        self.assertEqual(groundtruth, "<|sid_begin|>target<|sid_end|>")


if __name__ == "__main__":
    unittest.main()
