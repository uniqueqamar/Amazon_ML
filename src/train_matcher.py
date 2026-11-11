"""
Step 5: Train Pairwise Matcher Model
Mines hard negatives using candidate generation, computes rich similarity features,
trains a Histogram Gradient Boosted Decision Tree (LightGBM equivalent),
and optimizes the decision threshold against the competition's Macro F_0.5 metric.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config
from utils.features import compute_pair_features

def compute_entity_f05(predicted_dict: dict, ground_truth_dict: dict, all_s1_ids: list) -> float:
    """
    Computes Macro F_0.5 score as defined by the Amazon ML Challenge:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons:
      - correctly predicted empty list -> 1.0
      - predicted non-empty -> 0.0
    """
    scores = []
    for s1_id in all_s1_ids:
        preds = set(predicted_dict.get(s1_id, []))
        trues = set(ground_truth_dict.get(s1_id, []))

        if len(trues) == 0:
            scores.append(1.0 if len(preds) == 0 else 0.0)
            continue

        if len(preds) == 0:
            scores.append(0.0)
            continue

        tp = len(preds.intersection(trues))
        if tp == 0:
            scores.append(0.0)
            continue

        prec = tp / len(preds)
        rec = tp / len(trues)
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
        scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0

def build_training_dataset(df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame, df_train_gt: pd.DataFrame):
    """Generates balanced training set with positive pairs and hard negative candidate pairs."""
    print("Generating training dataset with hard negatives...")
    s1_dict = df_s1.set_index("id").to_dict(orient="index")
    s2_dict = df_s2.set_index("id").to_dict(orient="index")
    s3_dict = df_s3.set_index("id").to_dict(orient="index")

    known_positives = set(zip(df_train_gt["source_id"].astype(str), df_train_gt["s1_id"].astype(str)))

    pairs_list = []
    labels = []

    # 1. Positives
    for _, row in df_train_gt.iterrows():
        src_id = str(row["source_id"])
        s1_id = str(row["s1_id"])
        pair_type = str(row.get("source_pair", "s1_s2"))
        src_dict = s3_dict if "s3" in pair_type.lower() else s2_dict

        if src_id in src_dict and s1_id in s1_dict:
            src_rec = src_dict[src_id]
            s1_rec = s1_dict[s1_id]
            feats = compute_pair_features(
                name1=s1_rec.get("norm_name", ""),
                addr1=s1_rec.get("norm_address", ""),
                country1=s1_rec.get("country", ""),
                name2=src_rec.get("norm_name", ""),
                addr2=src_rec.get("norm_address", ""),
                country2=src_rec.get("country", ""),
                source_pair=pair_type,
                tfidf_score=1.0
            )
            pairs_list.append(feats)
            labels.append(1)

    n_pos = len(labels)
    print(f"Positive training pairs: {n_pos:,}")

    # 2. Hard Negatives via TF-IDF blocking
    train_queries = []
    for src_id in df_train_gt["source_id"].unique():
        s_id = str(src_id)
        if s_id in s2_dict:
            train_queries.append({"id": s_id, "clean_text": s2_dict[s_id].get("clean_text", ""), "source_pair": "s1_s2"})
        elif s_id in s3_dict:
            train_queries.append({"id": s_id, "clean_text": s3_dict[s_id].get("clean_text", ""), "source_pair": "s1_s3"})

    df_tq = pd.DataFrame(train_queries)
    if len(df_tq) > 0:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_features=50000)
        tfidf_s1 = vectorizer.fit_transform(df_s1["clean_text"])
        tfidf_tq = vectorizer.transform(df_tq["clean_text"])
        sim_mat = tfidf_tq.dot(tfidf_s1.T)

        s1_ids = df_s1["id"].values
        neg_count = 0
        target_negatives = n_pos * 4  # 4:1 hard negative ratio

        for r_idx in range(tfidf_tq.shape[0]):
            q_id = df_tq.iloc[r_idx]["id"]
            pair_type = df_tq.iloc[r_idx]["source_pair"]
            src_dict = s3_dict if pair_type == "s1_s3" else s2_dict
            src_rec = src_dict[q_id]

            row = sim_mat.getrow(r_idx)
            if row.nnz == 0:
                continue

            scores = row.data
            indices = row.indices
            sort_idx = np.argsort(-scores)

            for s_i in sort_idx[:15]:
                cand_s1_id = s1_ids[indices[s_i]]
                cand_score = scores[s_i]

                if (q_id, cand_s1_id) not in known_positives:
                    s1_rec = s1_dict[cand_s1_id]
                    feats = compute_pair_features(
                        name1=s1_rec.get("norm_name", ""),
                        addr1=s1_rec.get("norm_address", ""),
                        country1=s1_rec.get("country", ""),
                        name2=src_rec.get("norm_name", ""),
                        addr2=src_rec.get("norm_address", ""),
                        country2=src_rec.get("country", ""),
                        source_pair=pair_type,
                        tfidf_score=cand_score
                    )
                    pairs_list.append(feats)
                    labels.append(0)
                    neg_count += 1
                    if neg_count >= target_negatives:
                        break
            if neg_count >= target_negatives:
                break

    print(f"Hard negative pairs generated: {len(labels) - n_pos:,}")
    return pd.DataFrame(pairs_list), np.array(labels)

def evaluate_and_tune(model, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame, df_cand_val: pd.DataFrame, df_val_gt: pd.DataFrame):
    """Extracts features for validation candidates and tunes decision threshold on Macro F_0.5."""
    print("\nExtracting features for validation candidates...")
    s1_dict = df_s1.set_index("id").to_dict(orient="index")
    s2_dict = df_s2.set_index("id").to_dict(orient="index")
    s3_dict = df_s3.set_index("id").to_dict(orient="index")

    val_records = []
    feats_list = []

    for _, row in df_cand_val.iterrows():
        src_id = str(row["source_id"])
        s1_id = str(row["s1_id"])
        pair_type = str(row.get("source_pair", "s1_s2"))
        src_dict = s3_dict if "s3" in pair_type.lower() else s2_dict

        if src_id in src_dict and s1_id in s1_dict:
            src_rec = src_dict[src_id]
            s1_rec = s1_dict[s1_id]
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
            feats_list.append(f)
            val_records.append((s1_id, src_id))

    X_val = pd.DataFrame(feats_list)
    probs = model.predict_proba(X_val)[:, 1]

    # Build ground truth mapping per S1 entity
    gt_dict = {}
    for _, r in df_val_gt.iterrows():
        s1 = str(r["s1_id"])
        src = str(r["source_id"])
        gt_dict.setdefault(s1, []).append(src)

    all_val_s1 = list(df_val_gt["s1_id"].unique())

    best_f05 = -1.0
    best_thresh = 0.50

    for thresh in np.arange(0.20, 0.85, 0.05):
        pred_dict = {}
        for (s1_id, src_id), p in zip(val_records, probs):
            if p >= thresh:
                pred_dict.setdefault(s1_id, []).append(src_id)

        f05 = compute_entity_f05(pred_dict, gt_dict, all_val_s1)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    print(f"Optimal Decision Threshold: {best_thresh:.2f} (Macro F_0.5 = {best_f05:.4f})")
    return best_thresh, best_f05

def main():
    print("=" * 60)
    print(" Amazon ML Challenge: [Step 5] Train Matcher Model")
    print("=" * 60)

    df_s1 = pd.read_parquet(config.S1_VAL_NORMALIZED)
    df_s2 = pd.read_parquet(config.S2_VAL_NORMALIZED)
    df_s3 = pd.read_parquet(config.S3_VAL_NORMALIZED)
    df_train_gt = pd.read_parquet(config.GROUND_TRUTH_TRAIN)
    df_val_gt = pd.read_parquet(config.GROUND_TRUTH_VAL)
    df_cand_val = pd.read_csv(config.CANDIDATE_PAIRS_VAL)

    X_train, y_train = build_training_dataset(df_s1, df_s2, df_s3, df_train_gt)

    print("\nTraining Histogram Gradient Boosted Classifier...")
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    best_thresh, best_f05 = evaluate_and_tune(model, df_s1, df_s2, df_s3, df_cand_val, df_val_gt)

    model_path = config.MODELS_DIR / "matcher_model.joblib"
    meta_path = config.MODELS_DIR / "model_metadata.json"

    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "optimal_threshold": float(best_thresh),
            "validation_macro_f05": float(best_f05),
            "feature_names": list(X_train.columns)
        }, f, indent=2)

    print(f"\n✅ Saved trained model to {model_path}")
    print(f"✅ Saved model metadata to {meta_path}")
    print("✅ Step 5: Training complete!")

if __name__ == "__main__":
    main()
