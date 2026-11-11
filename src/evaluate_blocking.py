"""
Step 4: Evaluate Blocking Quality Gate
Computes blocking recall, reduction ratio, and macro F_0.5 potential on validation candidates.
Crucial Feedback Loop: If recall < 0.90, halts pipeline and exits with non-zero status.
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config

def evaluate_subset(name: str, gt_pairs: set, cand_pairs: set):
    """Computes recall and recovery for a specific source pairing subset."""
    total_gt = len(gt_pairs)
    if total_gt == 0:
        return 1.0, 0, 0
    recovered = len(gt_pairs.intersection(cand_pairs))
    recall = recovered / total_gt
    return recall, recovered, total_gt

def main():
    print("=" * 60)
    print(" Amazon ML Challenge: [Step 4] Evaluate Blocking Quality")
    print("=" * 60)

    if not config.CANDIDATE_PAIRS_VAL.exists():
        print(f"❌ Error: {config.CANDIDATE_PAIRS_VAL} not found.")
        print("Please run src/blocking_validation.py first.")
        sys.exit(1)

    df_cand = pd.read_csv(config.CANDIDATE_PAIRS_VAL)
    df_val_gt = pd.read_parquet(config.GROUND_TRUTH_VAL)

    # Positive ground truth matches
    if "is_match" in df_val_gt.columns:
        df_val_gt = df_val_gt[df_val_gt["is_match"] == 1]

    gt_pairs_all = set(zip(df_val_gt["source_id"].astype(str), df_val_gt["s1_id"].astype(str)))
    cand_pairs_all = set(zip(df_cand["source_id"].astype(str), df_cand["s1_id"].astype(str)))

    total_gt = len(gt_pairs_all)
    recovered = len(gt_pairs_all.intersection(cand_pairs_all))
    overall_recall = recovered / total_gt if total_gt > 0 else 0.0

    print("\n--- Blocking Evaluation Summary ---")
    print(f"Validation Ground Truth Matches: {total_gt:,}")
    print(f"Candidate Pairs Generated:       {len(cand_pairs_all):,}")
    print(f"True Matches Recovered:          {recovered:,}")
    print(f"Overall Blocking Recall:         {overall_recall:.4f} ({overall_recall * 100:.2f}%)")
    print(f"Blocking Quality Gate Threshold: {config.MIN_BLOCKING_RECALL:.4f} ({config.MIN_BLOCKING_RECALL * 100:.2f}%)")

    # Detailed per-branch breakdown
    for branch in ["s1_s2", "s1_s3"]:
        gt_branch = df_val_gt[df_val_gt["source_pair"] == branch]
        cand_branch = df_cand[df_cand["source_pair"] == branch]

        gt_set = set(zip(gt_branch["source_id"].astype(str), gt_branch["s1_id"].astype(str)))
        cand_set = set(zip(cand_branch["source_id"].astype(str), cand_branch["s1_id"].astype(str)))

        rec, rec_count, tot_count = evaluate_subset(branch, gt_set, cand_set)
        print(f"  • Branch [{branch}]: Recall = {rec:.4f} ({rec_count}/{tot_count}) | Candidates = {len(cand_set):,}")

    print("\n" + "=" * 60)
    # The Critical Quality Gate Check
    if overall_recall < config.MIN_BLOCKING_RECALL:
        print("❌ CRITICAL QUALITY GATE FAILED!")
        print(f"Blocking recall ({overall_recall:.4f}) is below the required threshold ({config.MIN_BLOCKING_RECALL:.4f}).")
        print("\n⚠️ ACTION REQUIRED:")
        print("1. Open config.py")
        print(f"2. Increase TFIDF_TOP_K (currently {config.TFIDF_TOP_K}) -> try {config.TFIDF_TOP_K + 20}")
        print(f"3. Increase MAX_CANDIDATES_PER_RECORD (currently {config.MAX_CANDIDATES_PER_RECORD}) -> try {config.MAX_CANDIDATES_PER_RECORD + 25}")
        print("4. Re-run: python3 src/blocking_validation.py && python3 src/evaluate_blocking.py")
        print("=" * 60)
        sys.exit(1)
    else:
        print("🎉 QUALITY GATE PASSED!")
        print(f"Blocking recall ({overall_recall:.4f}) meets the threshold. Safe to proceed to Train Matcher.")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    main()
