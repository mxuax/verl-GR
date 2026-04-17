#!/usr/bin/env python3
"""Train/Test Split Script

Randomly selects N samples from multiple parquet files as the test set, with remaining data as the training set.
Both datasets are shuffled before saving.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_all_parquet_files(file_paths: List[str], engine: str = 'pyarrow') -> pd.DataFrame:
    """Load and merge all parquet files.

    Args:
        file_paths: List of parquet file paths
        engine: Parquet engine, 'pyarrow' or 'fastparquet'

    Returns:
        Merged DataFrame
    """
    if not file_paths:
        logger.warning("No parquet files found")
        return pd.DataFrame()

    logger.info(f"Found {len(file_paths)} parquet files, starting to load...")

    dataframes = []
    for file_path in tqdm(file_paths, desc="Loading files"):
        try:
            df = pd.read_parquet(file_path, engine=engine)
            logger.debug(f"  Loaded {file_path}: {len(df)} rows")
            dataframes.append(df)
        except Exception as e:
            logger.error(f"  Failed to load {file_path}: {e}")
            continue

    if not dataframes:
        logger.warning("No files loaded successfully")
        return pd.DataFrame()

    # Merge all DataFrames
    logger.info("Merging all data...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Merge complete, total {len(combined_df)} rows")

    return combined_df


def split_train_test(
    df: pd.DataFrame,
    test_size: int,
    seed: int = None
) -> tuple:
    """Split DataFrame into training and test sets.

    Args:
        df: DataFrame to split
        test_size: Number of test samples
        seed: Random seed

    Returns:
        (train_df, test_df) tuple
    """
    if len(df) == 0:
        logger.warning("DataFrame is empty, cannot split")
        return pd.DataFrame(), pd.DataFrame()

    if test_size <= 0:
        raise ValueError(f"test_size must be greater than 0, current value: {test_size}")

    total_rows = len(df)

    if test_size >= total_rows:
        logger.warning(
            f"Test size ({test_size}) is greater than or equal to total rows ({total_rows}), "
            f"using all data as test set, training set will be empty"
        )
        return pd.DataFrame(), df.copy()

    # Use pandas sample method for random sampling, ensuring reproducibility
    if seed is not None:
        logger.info(f"Using random seed: {seed}")

    logger.info(f"Randomly selecting {test_size} rows from {total_rows} rows as test set...")

    # Use pandas sample method to randomly select test set
    test_df = df.sample(n=test_size, random_state=seed).copy()
    # Get test set indices
    test_indices = set(test_df.index)
    # Remaining data as training set
    train_df = df.drop(test_indices).copy()

    logger.info(f"Split complete: training set {len(train_df)} rows, test set {len(test_df)} rows")

    return train_df, test_df


def shuffle_dataframe(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """Shuffle DataFrame.

    Args:
        df: DataFrame to shuffle
        seed: Random seed (for reproducibility)

    Returns:
        Shuffled DataFrame
    """
    if len(df) == 0:
        return df.copy()

    # Use sample method for shuffling (frac=1 means sampling all data, i.e., shuffling)
    # random_state parameter ensures reproducibility
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return shuffled_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Randomly select N samples from multiple parquet files as test set, remaining data as training set'
    )
    parser.add_argument(
        '--input_files',
        type=str,
        nargs='+',
        required=True,
        help='List of input parquet file paths (can specify multiple files)'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        required=True,
        help='Number of test samples'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (optional, for reproducibility)'
    )
    parser.add_argument(
        '--engine',
        choices=['pyarrow', 'fastparquet'],
        default='pyarrow',
        help='Parquet processing engine (default: pyarrow)'
    )
    parser.add_argument(
        '--test_filename',
        type=str,
        default='test.parquet',
        help='Test set output filename (default: test.parquet)'
    )
    parser.add_argument(
        '--train_filename',
        type=str,
        default='train.parquet',
        help='Training set output filename (default: train.parquet)'
    )
    
    args = parser.parse_args()

    # Validate parameters
    if args.test_size <= 0:
        logger.error(f"test_size must be greater than 0, current value: {args.test_size}")
        sys.exit(1)
    
    # Validate input files exist
    input_files = []
    for file_path in args.input_files:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File does not exist, skipping: {file_path}")
            continue
        if not path.is_file():
            logger.warning(f"Path is not a file, skipping: {file_path}")
            continue
        if path.suffix.lower() != '.parquet':
            logger.warning(f"Not a parquet file, skipping: {file_path}")
            continue
        input_files.append(str(path))
    
    if not input_files:
        logger.error("No valid parquet files found")
        sys.exit(1)
    
    try:
        # 1. Load all parquet files
        logger.info("=" * 60)
        logger.info("Step 1: Loading parquet files...")
        combined_df = load_all_parquet_files(input_files, engine=args.engine)

        if len(combined_df) == 0:
            logger.error("No data loaded")
            sys.exit(1)
        
        # 2. Split training and test sets
        logger.info("=" * 60)
        logger.info("Step 2: Splitting training and test sets...")
        train_df, test_df = split_train_test(
            combined_df,
            test_size=args.test_size,
            seed=args.seed
        )
        
        if len(test_df) == 0:
            logger.error("Test set is empty, cannot continue")
            sys.exit(1)
        
        # 3. Shuffle data
        logger.info("=" * 60)
        logger.info("Step 3: Shuffling data...")
        
        # Use different seed offsets for training and test sets to ensure different shuffle results
        # If seed is provided, use different offsets; otherwise use None for both (completely random)
        train_seed = (args.seed + 1000) if args.seed is not None else None
        test_seed = (args.seed + 2000) if args.seed is not None else None
        
        logger.info("Shuffling training set...")
        train_df = shuffle_dataframe(train_df, seed=train_seed)

        logger.info("Shuffling test set...")
        test_df = shuffle_dataframe(test_df, seed=test_seed)
        
        # 4. Save results
        logger.info("=" * 60)
        logger.info("Step 4: Saving results...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        test_path = output_dir / args.test_filename
        train_path = output_dir / args.train_filename

        logger.info(f"Saving test set to: {test_path}")
        test_df.to_parquet(
            test_path,
            engine='pyarrow',
            index=False,
            compression='snappy'
        )
        
        if len(train_df) > 0:
            logger.info(f"Saving training set to: {train_path}")
            train_df.to_parquet(
                train_path,
                engine='pyarrow',
                index=False,
                compression='snappy'
            )
        else:
            logger.warning("Training set is empty, skipping save")
        
        # 5. Output statistics
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Number of input files: {len(input_files)}")
        logger.info(f"Original data rows: {len(combined_df)}")
        logger.info(f"Training set rows: {len(train_df)}")
        logger.info(f"Test set rows: {len(test_df)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Training set file: {train_path}")
        logger.info(f"Test set file: {test_path}")
        if args.seed is not None:
            logger.info(f"Random seed: {args.seed}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

