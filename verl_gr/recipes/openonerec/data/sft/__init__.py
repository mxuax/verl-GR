"""OpenOneRec SFT data preprocessing modules."""

# product_rec.py is the primary SFT data preparation script.
# It converts raw product recommendation metadata into chat-format parquet
# files with a ``messages`` column (JSON serialised), suitable for
# OneRecSFTDataset or MultiTurnSFTDataset.
