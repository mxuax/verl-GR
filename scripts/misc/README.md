# misc/

Ad-hoc utilities and alignment-debug tooling. These are **not** long-run production entry points — use the launchers in `scripts/` for training.

## rankgrpo_alignment/

Rank-GRPO ↔ TRL reference alignment and sidecar gating. Used by `RUN_DEBUG_STEP=30` in `run_rankgrpo.sh`.

| Script | Purpose |
| --- | --- |
| `export_trl_gate_sidecar_from_tb.py` | Export TRL logprob gate probes from TensorBoard → `rankgrpo_gate_sidecar.json`. **Called by `run_rankgrpo.sh`.** |
| `build_trl_gate_sidecar_from_train_log.py` | Build sidecar from TRL train-log console metrics (fallback when TB is sparse). |
| `compare_rankgrpo_tb.py` | Compare TensorBoard scalars between verl-gr and TRL runs. |
| `generate_rankgrpo_precision_report.py` | Offline precision report (fork TB vs TRL TB). |
| `run_offline_alignment_gate_report.py` | Offline per-step alignment gate report from fork TB vs `TRL_REF`. |

## checkpoint/

Self-contained checkpoint convert / merge / eval helpers (mainly MiniOneRec).

| Script | Purpose |
| --- | --- |
| `convert_ddp_to_hf.py` | Export DDP `model.pt` to HuggingFace layout. |
| `merge_fsdp_ckpt.py` | Merge FSDP shards into a single HF checkpoint. |
| `eval_compare_ckpts.py` | Beam-search eval and A/B checkpoint comparison. |

## sft_eval/

One-shot SFT checkpoint evaluation via the RL trainer's validation path.

| Script | Purpose |
| --- | --- |
| `eval_sft_minionerec.sh` | MiniOneRec constrained-beam eval. |
| `eval_sft_onerec.sh` | OpenOneRec two-stage eval. |

## profiling/

| Script | Purpose |
| --- | --- |
| `compare_nsys_nvtx.py` | Compare two `nsys stats --report nvtxsum` CSVs. |

## smoke/

| Script | Purpose |
| --- | --- |
| `smoke_test_main_4gpu.sh` | 4-GPU smoke test across minionerec / openonerec / rankgrpo launchers. |
