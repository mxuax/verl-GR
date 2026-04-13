# Contracts

`contracts/` defines the stable interfaces consumed by later phases.

## Included Contracts

- `tokenizer_contract.py`: tokenizer stage inputs, outputs, and artifact expectations
- `sft_contract.py`: SFT stage inputs and outputs
- `distill_contract.py`: distillation stage inputs and outputs
- `rl_contract.py`: RL stage inputs, outputs, and reward entrypoints
- `eval_contract.py`: evaluation stage inputs and reports
- `task_composition.py`: allowed stage composition at recipe level

## Rules

- Contracts describe interfaces, not implementations.
- Contracts must remain backend-agnostic during Phase A.
- Later phases may extend contracts, but should not silently redefine them.
