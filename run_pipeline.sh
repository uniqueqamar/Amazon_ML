#!/usr/bin/env bash
# Amazon ML Challenge 2026: Business Entity Resolution End-to-End Pipeline
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo " Amazon ML Challenge 2026: Business Entity Resolution Pipeline"
echo " Directory: $SCRIPT_DIR"
echo "================================================================="

# Step 1: Normalize Data
echo ""
echo "[Step 1/8] Ingesting and Normalizing Business Entity Records..."
python3 src/build_normalized_data.py

# Step 2: Validation Split
echo ""
echo "[Step 2/8] Creating Stratified Leak-Free Validation Split..."
python3 src/validation_split.py

# Step 3: Block Validation Set (S1xS2 and S1xS3)
echo ""
echo "[Step 3/8] Running Candidate Generation on Validation Set..."
python3 src/blocking_validation.py

# Step 4: Evaluate Blocking Quality Gate (Halts if Recall < 0.90)
echo ""
echo "[Step 4/8] Evaluating Blocking Recall (Quality Gate)..."
if ! python3 src/evaluate_blocking.py; then
    echo ""
    echo "================================================================="
    echo "⛔ PIPELINE STOPPED AT QUALITY GATE!"
    echo "Blocking recall is below 0.90."
    echo "Matcher training was skipped to prevent poor downstream matching."
    echo "Please update TFIDF_TOP_K in config.py and re-run this pipeline."
    echo "================================================================="
    exit 1
fi

# Step 5: Train Pairwise Matcher Model (HistGradientBoosting / LightGBM)
echo ""
echo "[Step 5/8] Training Matcher Model (Optimizing Macro F_0.5)..."
python3 src/train_matcher.py

# Step 6: Block Full Set (Generates official candidate_pairs.tsv)
echo ""
echo "[Step 6/8] Generating Candidate Pairs (candidate_pairs.tsv)..."
# If arguments were provided to run_pipeline.sh (e.g. --split test or test), pass them along; otherwise default to val
if [ $# -gt 0 ]; then
    python3 src/blocking_full.py "$@"
else
    python3 src/blocking_full.py --split val
fi

# Step 7: Predict Matches (Generates official matching_results.tsv)
echo ""
echo "[Step 7/8] Scoring Candidates and Generating Final Match Predictions..."
python3 src/predict.py

# Step 8: Validate Submission Against Amazon ML Rules
echo ""
echo "[Step 8/8] Validating Final Submission Files Against Competition Rules..."
python3 utils/validate_submission.py

echo ""
echo "================================================================="
echo "🎉 PIPELINE COMPLETE! Verified submission files ready in output/"
echo "================================================================="
