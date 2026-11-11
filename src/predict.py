"""
Step 7: Predict Matches
Loads candidate pairs and the trained matcher model.
Extracts pairwise features, scores candidates, applies optimal Macro F_0.5 decision threshold,
and writes the official competition submission:
  output/matching_results.tsv
Columns: source1_entity_id, matched_entity_ids
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config
from utils.features import compute_pair_features

def main():
    print("=" * 60)
    print(" Amazon ML Challenge: [Step 7] Predict Matches")
    print("=" * 60)

    cand_tsv = config.CANDIDATE_PAIRS_TEST
    interm_parquet = config.PROCESSED_DIR / "scored_candidates_interm.parquet"
    model_path = config.MODELS_DIR / "matcher_model.joblib"
    meta_path = config.MODELS_DIR / "model_metadata.json"

    if not cand_tsv.exists():
        print(f"❌ Error: {cand_tsv} not found. Run blocking_full.py first.")
        sys.exit(1)

    if not model_path.exists() or not meta_path.exists():
        print(f"❌ Error: Model not found at {model_path}. Run train_matcher.py first.")
        sys.exit(1)

    df_cand_meta = pd.read_csv(cand_tsv, sep='\t')
    all_s1_ids = df_cand_meta["source1_entity_id"].astype(str).tolist()

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    threshold = meta.get("optimal_threshold", config.MATCH_THRESHOLD)
    print(f"Loaded trained model. Optimal decision threshold: {threshold:.2f}")

    # Load normalized entities for fast lookup
    df_s1 = pd.read_parquet(config.S1_VAL_NORMALIZED)
    df_s2 = pd.read_parquet(config.S2_VAL_NORMALIZED)
    df_s3 = pd.read_parquet(config.S3_VAL_NORMALIZED)

    s1_dict = df_s1.set_index("id").to_dict(orient="index")
    s2_dict = df_s2.set_index("id").to_dict(orient="index")
    s3_dict = df_s3.set_index("id").to_dict(orient="index")

    if interm_parquet.exists():
        df_pairs = pd.read_parquet(interm_parquet)
    else:
        # Reconstruct from TSV
        unrolled = []
        for _, r in df_cand_meta.iterrows():
            s1 = str(r["source1_entity_id"])
            c_str = str(r["candidate_entity_ids"]) if pd.notnull(r["candidate_entity_ids"]) else ""
            for c_id in c_str.split(","):
                c_id = c_id.strip()
                if c_id:
                    pair_type = "s1_s3" if c_id.startswith("S3") else "s1_s2"
                    unrolled.append({"s1_id": s1, "source_id": c_id, "tfidf_score": 0.5, "source_pair": pair_type})
        df_pairs = pd.DataFrame(unrolled)

    print(f"Extracting pairwise similarity features for {len(df_pairs):,} candidate pairs...")
    feats = []
    pair_tuples = []

    for _, row in df_pairs.iterrows():
        s1_id = str(row["s1_id"])
        src_id = str(row["source_id"])
        pair_type = str(row.get("source_pair", "s1_s2"))

        src_dict = s3_dict if "s3" in pair_type.lower() else s2_dict
        s1_rec = s1_dict.get(s1_id, {})
        src_rec = src_dict.get(src_id, {})

        f = compute_pair_features(
            name1=s1_rec.get("norm_name", ""),
            addr1=s1_rec.get("norm_address", ""),
            country1=s1_rec.get("country", ""),
            name2=src_rec.get("norm_name", ""),
            addr2=src_rec.get("norm_address", ""),
            country2=src_rec.get("country", ""),
            source_pair=pair_type,
            tfidf_score=row.get("tfidf_score", 0.0)
        )
        feats.append(f)
        pair_tuples.append((s1_id, src_id))

    if len(feats) > 0:
        X = pd.DataFrame(feats)
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = np.array([])

    # Group predicted matches above threshold
    matches_dict = {}
    for (s1_id, src_id), p in zip(pair_tuples, probs):
        if p >= threshold:
            matches_dict.setdefault(s1_id, []).append(src_id)

    # Format official submission rows
    submission_rows = []
    matched_count = 0
    singleton_count = 0

    for s1_id in all_s1_ids:
        matches = matches_dict.get(s1_id, [])
        # Unique preserving order
        seen = set()
        uniq_matches = [m for m in matches if not (m in seen or seen.add(m))]

        if len(uniq_matches) > 0:
            matched_count += 1
            matched_str = ",".join(uniq_matches)
        else:
            singleton_count += 1
            matched_str = ""

        submission_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matched_str
        })

    df_submission = pd.DataFrame(submission_rows)
    out_file = config.MATCHING_RESULTS_TEST
    df_submission.to_csv(out_file, sep='\t', index=False)

    print(f"\n✅ Predictions saved to: {out_file}")
    print(f"   Total Source 1 entities: {len(df_submission):,}")
    print(f"   Entities with matches:   {matched_count:,}")
    print(f"   Identified singletons:   {singleton_count:,}")
    print("✅ Step 7: Prediction complete!")

if __name__ == "__main__":
    main()
