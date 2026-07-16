"""SFT evaluation — reuse verl-GR RL rollout infrastructure for eval.

Reuses the existing RL trainer's validation path with ``val_before_train=true``
and ``total_training_steps=0`` so that the SFT checkpoint is loaded, the
constrained-beam / two-stage rollout runs, and hit-rate metrics are collected,
then exits without any training.

Usage (via shell script)::

    bash scripts/misc/sft_eval/eval_sft_minionerec.sh /path/to/sft_ckpt
"""

# The eval is implemented by invoking the RL trainer with eval-only
# configuration overrides.  See ``scripts/misc/sft_eval/eval_sft_*.sh`` for the
# canonical invocation.

from __future__ import annotations
