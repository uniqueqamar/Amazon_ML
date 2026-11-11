"""
Step 6: Block Full Set (Test or Validation)
Runs candidate generation across both branches:
  1) Source 1 x Source 2
  2) Source 1 x Source 3
Groups candidates per Source 1 entity into the official submission TSV format:
  output/candidate_pairs.tsv
Columns: source1_entity_id, candidate_entity_ids
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config
from utils.normalization import normalize_business_name, normalize_address, create_entity_representation

def load_and_prep(filepath: Path, name: str, limit: int = None):
    """Loads CSV or TSV file, applies text normalization and creates clean representation."""
    sep = '\t' if filepath.suffix == '.tsv' else ','
    print(f"Loading {name} from {filepath.name}...")
    df = pd.read_csv(filepath, sep=sep, nrows=limit)

    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "entity_id" in cl or cl == "id":
            col_map["id"] = c
        elif any(k in cl for k in ["name", "title"]):
            col_map["name"] = c
        elif any(k in cl for k in ["address", "loc"]):
            col_map["address"] = c
        elif "country" in cl:
            col_map["country"] = c

    res = pd.DataFrame()
    res["id"] = df[col_map.get("id", df.columns[0])].astype(str)
    res["name"] = df[col_map.get("name", df.columns[1])].fillna("").astype(str)
    res["address"] = df[col_map.get("address", df.columns[2])].fillna("").astype(str) if len(df.columns) > 2 else ""
    res["country"] = df[col_map["country"]].fillna("").astype(str) if "country" in col_map else ""

    res["clean_text"] = [
        create_entity_representation(n, a, c)
        for n, a, c in zip(res["name"], res["address"], res["country"])
    ]
    return res

def block_branch(df_s1: pd.DataFrame, df_queries: pd.DataFrame, branch_name: str, top_k: int):
    """Fits TF-IDF on S1 and generates candidate pairs for query records."""
    print(f"[{branch_name}] Indexing {len(df_s1):,} S1 entities...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_features=config.TFIDF_MAX_FEATURES
    )

    tfidf_s1 = vectorizer.fit_transform(df_s1["clean_text"])
    tfidf_queries = vectorizer.transform(df_queries["clean_text"])

    print(f"[{branch_name}] Computing cosine similarities for {len(df_queries):,} records...")
    sim_matrix = tfidf_queries.dot(tfidf_s1.T)

    s1_ids = df_s1["id"].values
    s1_countries = df_s1["country"].str.lower().str.strip().values
    query_ids = df_queries["id"].values
    query_countries = df_queries["country"].str.lower().str.strip().values

    pairs = []
    for r_idx in range(tfidf_queries.shape[0]):
        row = sim_matrix.getrow(r_idx)
        if row.nnz == 0:
            continue
        scores = row.data
        indices = row.indices
        q_c = query_countries[r_idx]

        if q_c:
            mask = [not (s1_countries[idx] and s1_countries[idx] != q_c) for idx in indices]
            scores = scores[mask]
            indices = indices[mask]

        if len(scores) == 0:
            continue

        if len(scores) > top_k:
            best_idx = np.argpartition(scores, -top_k)[-top_k:]
            best_scores = scores[best_idx]
            best_s1_idx = indices[best_idx]
        else:
            best_scores = scores
            best_s1_idx = indices

        sort_o = np.argsort(-best_scores)
        q_id = query_ids[r_idx]

        for s_i in sort_o[:config.MAX_CANDIDATES_PER_RECORD]:
            pairs.append((s1_ids[best_s1_idx[s_i]], q_id, float(best_scores[s_i])))

    return pairs

def main():
    parser = argparse.ArgumentParser(description="Full blocking across test or validation records")
    parser.add_argument("--split", type=str, default=None, choices=["test", "val"], help="Target split (test or val)")
    parser.add_argument("positional_split", nargs="?", choices=["test", "val"], default=None, help="Optional positional split (test or val)")
    args = parser.parse_args()

    split_target = args.split or args.positional_split or "test"

    print("=" * 60)
    print(f" Amazon ML Challenge: [Step 6] Block Full Set (Split: {split_target})")
    print("=" * 60)

    if split_target == "val" or not config.TEST_S1_TSV.exists():
        print("Using validation dataset for blocking...")
        df_s1 = load_and_prep(config.S1_VAL_CSV, "S1_Val")
        df_s2 = load_and_prep(config.S2_VAL_CSV, "S2_Val")
        df_s3 = load_and_prep(config.S3_VAL_CSV, "S3_Val")
    else:
        print("Using full test dataset...")
        df_s1 = load_and_prep(config.TEST_S1_TSV, "Test_S1")
        df_s2 = load_and_prep(config.TEST_S2_TSV, "Test_S2")
        df_s3 = load_and_prep(config.TEST_S3_TSV, "Test_S3")

    # Run Branch 1: S1 x S2
    pairs_s1_s2 = block_branch(df_s1, df_s2, "s1_s2", config.TFIDF_TOP_K)

    # Run Branch 2: S1 x S3
    pairs_s1_s3 = block_branch(df_s1, df_s3, "s1_s3", config.TFIDF_TOP_K)

    all_pairs = pairs_s1_s2 + pairs_s1_s3
    print(f"\nTotal raw candidate pairs found: {len(all_pairs):,}")

    # Also save tabular pairs for predict.py to score
    df_raw_pairs = pd.DataFrame(all_pairs, columns=["s1_id", "source_id", "tfidf_score"])
    df_raw_pairs["source_pair"] = df_raw_pairs["source_id"].apply(lambda x: "s1_s3" if str(x).startswith("S3") else "s1_s2")
    df_raw_pairs.to_parquet(config.PROCESSED_DIR / "scored_candidates_interm.parquet", index=False)

    # Group by source1_entity_id as required by the Amazon ML submission specification
    cand_dict = {}
    for s1_id, src_id, _ in all_pairs:
        cand_dict.setdefault(s1_id, []).append(src_id)

    submission_rows = []
    for s1_id in df_s1["id"]:
        cands = cand_dict.get(s1_id, [])
        # Unique and preserve order
        seen = set()
        uniq_cands = [c for c in cands if not (c in seen or seen.add(c))]
        submission_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(uniq_cands)
        })

    df_submission = pd.DataFrame(submission_rows)
    out_file = config.CANDIDATE_PAIRS_TEST
    df_submission.to_csv(out_file, sep='\t', index=False)

    print(f"\n✅ Official candidate pairs saved to: {out_file}")
    print(f"   Total Source 1 entities: {len(df_submission):,}")
    print("✅ Step 6: Full blocking complete!")

if __name__ == "__main__":
    main()
