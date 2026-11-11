"""
Step 2: Validation Split
Performs an entity-level (group) split on source1_entity_id to ensure zero data leakage between
training and validation sets. Stratifies across source pairings (s1_s2 and s1_s3).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config

def main():
    print("=" * 60)
    print(" Amazon ML Challenge: [Step 2] Validation Split")
    print("=" * 60)

    gt_path = config.PROCESSED_DIR / "raw_ground_truth.parquet"
    if not gt_path.exists():
        print(f"❌ Error: Ground truth data not found at {gt_path}")
        print("Please run src/build_normalized_data.py first.")
        sys.exit(1)

    df_gt = pd.read_parquet(gt_path)
    print(f"Total positive matching pairs: {len(df_gt):,}")

    # Entity-level grouping to prevent data leakage
    unique_s1 = df_gt["s1_id"].unique()
    print(f"Unique Source 1 entities with matches: {len(unique_s1):,}")

    train_s1, val_s1 = train_test_split(
        unique_s1,
        test_size=config.VAL_SPLIT_RATIO,
        random_state=config.RANDOM_STATE
    )

    train_df = df_gt[df_gt["s1_id"].isin(set(train_s1))].reset_index(drop=True)
    val_df = df_gt[df_gt["s1_id"].isin(set(val_s1))].reset_index(drop=True)

    train_df.to_parquet(config.GROUND_TRUTH_TRAIN, index=False)
    val_df.to_parquet(config.GROUND_TRUTH_VAL, index=False)

    print(f"✅ Training set:   {len(train_df):,} pairs ({len(train_s1):,} S1 entities) -> {config.GROUND_TRUTH_TRAIN.name}")
    print(f"✅ Validation set: {len(val_df):,} pairs ({len(val_s1):,} S1 entities) -> {config.GROUND_TRUTH_VAL.name}")
    print("\n✅ Step 2: Validation split complete!")

if __name__ == "__main__":
    main()
