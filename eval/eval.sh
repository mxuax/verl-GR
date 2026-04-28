python eval/openonerec_eval.py \
  --checkpoint-root /YOUR/CKPT_ROOT  \
  --test-max-sample -1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --test-parquet /YOUR/PARQUET_DIR