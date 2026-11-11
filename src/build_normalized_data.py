"""
Step 1: Build Normalized Data
Ingests business entity records from validation/ (s1_val.csv, s2_val.csv, s3_val.csv, gt_val.csv).
Applies legal abbreviation standardization, location/state mapping, accent stripping,
and generates clean search representations for TF-IDF indexing.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config
from utils.normalization import (
    normalize_business_name,
    normalize_address,
    create_entity_representation
)

def normalize_source_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Standardizes columns and applies text normalization to a source DataFrame."""
    col_map = {}
    for col in df.columns:
        c_low = col.lower().strip()
        if c_low in ["entity_id", "id", f"{source_name.lower()}_id"]:
            col_map["id"] = col
        elif any(k in c_low for k in ["name", "business_name", "title"]):
            col_map["name"] = col
        elif any(k in c_low for k in ["address", "business_address", "location"]):
            col_map["address"] = col
        elif "country" in c_low:
            col_map["country"] = col

    id_col = col_map.get("id", df.columns[0])
    name_col = col_map.get("name", df.columns[1] if len(df.columns) > 1 else df.columns[0])
    addr_col = col_map.get("address", df.columns[2] if len(df.columns) > 2 else name_col)
    country_col = col_map.get("country", None)

    print(f"[{source_name}] Using columns -> ID: '{id_col}', Name: '{name_col}', Addr: '{addr_col}', Country: '{country_col}'")

    res = pd.DataFrame()
    res["id"] = df[id_col].astype(str)
    res["raw_name"] = df[name_col].fillna("").astype(str)
    res["raw_address"] = df[addr_col].fillna("").astype(str) if addr_col != name_col else ""
    res["country"] = df[country_col].fillna("").astype(str) if country_col else ""

    print(f"[{source_name}] Normalizing text fields for {len(res):,} records...")
    res["norm_name"] = res["raw_name"].apply(normalize_business_name)
    res["norm_address"] = res["raw_address"].apply(normalize_address)
    res["clean_text"] = [
        create_entity_representation(n, a, c)
        for n, a, c in zip(res["norm_name"], res["norm_address"], res["country"])
    ]
    return res

def unroll_ground_truth(df_gt: pd.DataFrame) -> pd.DataFrame:
    """
    Unrolls ground truth records:
    Input: source1_entity_id, matched_entity_ids (comma-separated list)
    Output: s1_id, source_id, source_pair ('s1_s2' or 's1_s3'), is_match
    """
    cols = [c.lower() for c in df_gt.columns]
    s1_col = next((df_gt.columns[i] for i, c in enumerate(cols) if "s1" in c or "source1" in c), df_gt.columns[0])
    match_col = next((df_gt.columns[i] for i, c in enumerate(cols) if any(x in c for x in ["matched", "target", "s2", "s3"])), df_gt.columns[1])

    rows = []
    for _, r in df_gt.iterrows():
        s1_id = str(r[s1_col]).strip()
        matched = str(r[match_col]).strip() if pd.notnull(r[match_col]) else ""

        if not matched or matched.lower() == "nan":
            # Singleton (no match)
            continue

        for m_id in matched.split(","):
            m_id = m_id.strip()
            if not m_id:
                continue
            pair_type = "s1_s3" if m_id.upper().startswith("S3") else "s1_s2"
            rows.append({
                "s1_id": s1_id,
                "source_id": m_id,
                "source_pair": pair_type,
                "is_match": 1
            })

    return pd.DataFrame(rows)

def main():
    print("=" * 60)
    print(" Amazon ML Challenge: [Step 1] Build Normalized Data")
    print("=" * 60)

    # 1. Load validation data files
    if not config.S1_VAL_CSV.exists() or not config.S2_VAL_CSV.exists() or not config.S3_VAL_CSV.exists():
        print(f"❌ Error: Validation source files not found in {config.VALIDATION_DIR}")
        sys.exit(1)

    print("Loading validation datasets...")
    df_s1 = pd.read_csv(config.S1_VAL_CSV)
    df_s2 = pd.read_csv(config.S2_VAL_CSV)
    df_s3 = pd.read_csv(config.S3_VAL_CSV)

    norm_s1 = normalize_source_df(df_s1, "Source1")
    norm_s2 = normalize_source_df(df_s2, "Source2")
    norm_s3 = normalize_source_df(df_s3, "Source3")

    norm_s1.to_parquet(config.S1_VAL_NORMALIZED, index=False)
    norm_s2.to_parquet(config.S2_VAL_NORMALIZED, index=False)
    norm_s3.to_parquet(config.S3_VAL_NORMALIZED, index=False)

    print(f"✅ Saved normalized S1: {len(norm_s1):,} records -> {config.S1_VAL_NORMALIZED.name}")
    print(f"✅ Saved normalized S2: {len(norm_s2):,} records -> {config.S2_VAL_NORMALIZED.name}")
    print(f"✅ Saved normalized S3: {len(norm_s3):,} records -> {config.S3_VAL_NORMALIZED.name}")

    # 2. Process Ground Truth
    if config.GT_VAL_CSV.exists():
        df_gt_raw = pd.read_csv(config.GT_VAL_CSV)
        unrolled_gt = unroll_ground_truth(df_gt_raw)
        gt_out = config.PROCESSED_DIR / "raw_ground_truth.parquet"
        unrolled_gt.to_parquet(gt_out, index=False)
        print(f"✅ Unrolled ground truth: {len(unrolled_gt):,} positive matching pairs -> {gt_out.name}")

    print("\n✅ Step 1: Normalization complete!")

if __name__ == "__main__":
    main()
