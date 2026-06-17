# verl-GR Tests

Tests are organized by recipe and concern:

```
tests/
├── smoke/          # Hydra compose (+ optional training startup)
├── core/           # Shared entrypoint / task selection
├── minionerec/     # MiniOneRec parity, reward, logprob, LoRA
├── openonerec/     # OpenOneRec rollout + eval helpers
└── rankgrpo/       # Rank-GRPO loss / mask math
```

## Run

```bash
# All CPU-safe tests
pytest tests/ -m "not gpu"

# Single recipe
pytest tests/minionerec/

# Hydra compose smoke (all recipes)
pytest tests/smoke/test_startup.py::test_hydra_compose

# Optional full training startup (needs GPU, model, parquet data)
VERL_GR_SMOKE_TRAIN=1 pytest tests/smoke/ -m gpu
```

Loss convergence and full parity are validated outside this suite.
