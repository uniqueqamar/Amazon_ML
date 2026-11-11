"""
Step 3: Blocking Validation Set
Executes candidate generation twice internally:
  1) Source 1 x Source 2
  2) Source 1 x Source 3
Extracts Top-K candidate matches using character n-gram TF-IDF cosine similarity,
merges both branches, and writes to processed/candidate_pairs_val.csv.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config

def block_pairings(df_s1: pd.DataFrame, df_queries: pd.DataFrame, source_pair: str, top_k: int):
    """
    Fits TF-IDF on S1 and finds top-k cosine similarity matches for query records (S2 or S3).
    Respects country compatibility when both are present.
    """
    print(f"[{source_pair}] Building TF-IDF index for {len(df_s1):,} S1 reference targets...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_features=config.TFIDF_MAX_FEATURES
    )

    tfidf_s1 = vectorizer.fit_transform(df_s1["clean_text"])
    print(f"[{source_pair}] Transforming {len(df_queries):,} query records...")
    tfidf_queries = vectorizer.transform(df_queries["clean_text"])

    print(f"[{source_pair}] Computing dot product cosine similarities...")
    sim_matrix = tfidf_queries.dot(tfidf_s1.T)

    candidates = []
    s1_ids = df_s1["id"].values
    s1_countries = df_s1["country"].str.lower().str.strip().values
    query_ids = df_queries["id"].values
    query_countries = df_queries["country"].str.lower().str.strip().values

    for row_idx in range(tfidf_queries.shape[0]):
        row = sim_matrix.getrow(row_idx)
        if row.nnz == 0:
            continue
        scores = row.data
        indices = row.indices
        q_country = query_countries[row_idx]

        # Country filter: skip candidate if both have countries and they mismatch
        if q_country:
            valid_mask = [
                not (s1_countries[t_idx] and s1_countries[t_idx] != q_country)
                for t_idx in indices
            ]
            scores = scores[valid_mask]
            indices = indices[valid_mask]

        if len(scores) == 0:
            continue

        if len(scores) > top_k:
            best_idx = np.argpartition(scores, -top_k)[-top_k:]
            best_scores = scores[best_idx]
            best_s1_indices = indices[best_idx]
        else:
            best_scores = scores
            best_s1_indices = indices

        sort_order = np.argsort(-best_scores)
        q_id = query_ids[row_idx]

        for rank, s_idx in enumerate(sort_order[:config.MAX_CANDIDATES_PER_RECORD]):
            candidates.append({
                "source_id": q_id,
                "s1_id": s1_ids[best_s1_indices[s_idx]],
                "tfidf_score": float(best_scores[s_idx]),
                "source_pair": source_pair,
                "rank": rank + 1
            })

    df_cand = pd.DataFrame(candidates)
    print(f"[{source_pair}] Generated {len(df_cand):,} candidate pairs.")
    return df_cand

def main():
    print("=" * 60)
    print(" Amazon ML Challenge: [Step 3] Block Validation Set")
    print(f" Config: TFIDF_TOP_K={config.TFIDF_TOP_K}, MAX_CANDIDATES={config.MAX_CANDIDATES_PER_RECORD}")
    print("=" * 60)

    # Load normalized validation datasets
    df_s1 = pd.read_parquet(config.S1_VAL_NORMALIZED)
    df_s2 = pd.read_parquet(config.S2_VAL_NORMALIZED)
    df_s3 = pd.read_parquet(config.S3_VAL_NORMALIZED)

    # Branch 1: S1 x S2
    cand_s1_s2 = block_pairings(df_s1, df_s2, "s1_s2", config.TFIDF_TOP_K)

    # Branch 2: S1 x S3
    cand_s1_s3 = block_pairings(df_s1, df_s3, "s1_s3", config.TFIDF_TOP_K)

    # Merge candidates
    merged_val_cand = pd.concat([cand_s1_s2, cand_s1_s3], ignore_index=True)
    merged_val_cand.to_csv(config.CANDIDATE_PAIRS_VAL, index=False)

    print(f"\n✅ Merged candidate validation pairs saved to {config.CANDIDATE_PAIRS_VAL.name}")
    print(f"   Total validation candidates: {len(merged_val_cand):,}")
    print("✅ Step 3: Validation blocking complete!")

if __name__ == "__main__":
    main()
