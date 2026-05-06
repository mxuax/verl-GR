"""Task-specific trainer adapters aligned with verl trainer override points."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class TrainerTaskAdapter:
    """Default no-op adapter for task-specific trainer behavior."""

    def prepare_gen_batch(self, trainer, batch: DataProto) -> DataProto:
        return RayPPOTrainer._get_gen_batch(trainer, batch)

    def validate(self, trainer):
        return RayPPOTrainer._validate(trainer)

    def dump_generations(
        self,
        trainer,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
        ground_truths=None,
    ):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{trainer.global_steps}.jsonl")
        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [trainer.global_steps] * n,
        }
        if ground_truths and len(ground_truths) == n:
            base_data["ground_truth"] = ground_truths
        for key, values in reward_extra_infos_dict.items():
            if len(values) == n:
                base_data[key] = values
        lines = []
        for i in range(n):
            lines.append(json.dumps({key: value[i] for key, value in base_data.items()}, ensure_ascii=False, default=str))
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

    def maybe_log_val_generations(self, trainer, inputs, outputs, scores):
        generations_to_log = trainer.config.trainer.get("log_val_generations", 0)
        if generations_to_log == 0:
            return
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda item: item[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        trainer.validation_generations_logger.log(
            trainer.config.trainer.logger,
            samples[:generations_to_log],
            trainer.global_steps,
        )

    def postprocess_rewards(
        self,
        trainer,
        batch: DataProto,
        reward_batch: DataProto,
    ) -> tuple[DataProto, dict[str, Any]]:
        """Optionally rewrite rm_scores before advantage computation."""

        return reward_batch, {}

    def evaluate_and_prune_checkpoint(self, trainer, local_global_step_folder: str, metrics=None) -> None:
        """Optionally run task-specific checkpoint pruning/evaluation."""

        return None
