# Benchmark launchers

Time-boxed full-training runs for loss + wall-clock comparison.

| Script | Env | Model |
|---|---|---|
| `run_openonerec_benchmark.sh` | `conda activate vllm-gr` | OneRec-1.7B-pro snapshot |
| `run_minionerec_benchmark.sh` | `conda activate MiniOneRec` | checkpoint-390 |

```bash
cd /home/fq9hpsac/fq9hpsacuser04/workspace/verl-GR
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/benchmark/run_openonerec_benchmark.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/benchmark/run_minionerec_benchmark.sh
```

Optional overrides:

```bash
TRAIN_MAX_SAMPLES=1000 VAL_MAX_SAMPLES=100 bash scripts/benchmark/run_openonerec_benchmark.sh
TRAIN_MAX_SAMPLES=4096 bash scripts/benchmark/run_minionerec_benchmark.sh
```

Wall clock is written to `outputs/<experiment>/benchmark_wallclock.txt`.

Full plan: [docs/verl_gr/full_training_benchmark.md](../../docs/verl_gr/full_training_benchmark.md).
