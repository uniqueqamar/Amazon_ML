# Amazon ML — Business Entity Resolution

A Python pipeline for matching noisy business records from Source 2 and Source 3 to a deduplicated Source 1 reference.

## Approach

The pipeline normalizes business names and addresses, generates candidates using character n-gram TF-IDF, checks blocking recall, trains a pairwise matcher, selects a matching threshold, and exports candidate and match TSV files. A submission validator and ZIP packaging utility are included.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional accelerators can be installed separately with `pip install rapidfuzz lightgbm`.

## Data and execution

Datasets and trained models are not included. Paths are configured in `src/config.py`. Before running the pipeline, supply the prepared validation files `validation/s1_val.csv`, `validation/s2_val.csv`, `validation/s3_val.csv`, and `validation/gt_val.csv` with the columns expected by the ingestion script. For test prediction, also supply `dataset/test/test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv`.

```bash
# Full pipeline using the prepared validation inputs
bash run_pipeline.sh

# Train on validation inputs and generate predictions for test inputs
bash run_pipeline.sh --split test
```

Generated files are written to `output/candidate_pairs.tsv` and `output/matching_results.tsv`. Intermediate data and model artifacts are stored locally in `processed/` and `models/`.

## Repository contents

- `src/`: configuration, normalization ingestion, validation splitting, blocking, matcher training, and prediction.
- `utils/`: normalization and similarity features, submission validation, and ZIP packaging.
- `run_pipeline.sh`: pipeline entry point.
- `requirements.txt`: dependencies supplied with the project.
- `docs/problem_statement.md`: original challenge README supplied with the submission.
- `Documentation_template.md`: supplied methodology template; placeholders remain unfilled.

The uploaded source is preserved. Python syntax and shell syntax were checked during import; full training and evaluation require the datasets and have not been rerun. No benchmark results are claimed here.
