start_time=$(date +%s)
echo "Eval started at: $(date '+%Y-%m-%d %H:%M:%S')"

python eval/openonerec_eval.py \
  --checkpoint-root /YOUR/CKPT_ROOT  \
  --test-max-sample -1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --test-parquet /YOUR/PARQUET_DIR
status=$?

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf 'Eval finished at: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
printf 'Eval elapsed time: %02d:%02d:%02d (%d seconds)\n' \
  $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60)) "$elapsed"

exit "$status"