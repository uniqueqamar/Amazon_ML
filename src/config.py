"""
Configuration for Amazon ML Challenge: Business Entity Resolution Pipeline
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DATA_DIR = DATASET_DIR / "train"
TEST_DATA_DIR = DATASET_DIR / "test"
VALIDATION_DIR = PROJECT_ROOT / "validation"
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

for dir_path in [PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Validation source files
S1_VAL_CSV = VALIDATION_DIR / "s1_val.csv"
S2_VAL_CSV = VALIDATION_DIR / "s2_val.csv"
S3_VAL_CSV = VALIDATION_DIR / "s3_val.csv"
GT_VAL_CSV = VALIDATION_DIR / "gt_val.csv"

# Test source files
TEST_S1_TSV = TEST_DATA_DIR / "test_source1.tsv"
TEST_S2_TSV = TEST_DATA_DIR / "test_source2.tsv"
TEST_S3_TSV = TEST_DATA_DIR / "test_source3.tsv"

# Train source files
TRAIN_S1_TSV = TRAIN_DATA_DIR / "train_source1.tsv"
TRAIN_S2_TSV = TRAIN_DATA_DIR / "train_source2.tsv"
TRAIN_S3_TSV = TRAIN_DATA_DIR / "train_source3.tsv"
TRAIN_GT_TSV = TRAIN_DATA_DIR / "train_ground_truth.tsv"

# Processed intermediate data files
S1_VAL_NORMALIZED = PROCESSED_DIR / "s1_val_normalized.parquet"
S2_VAL_NORMALIZED = PROCESSED_DIR / "s2_val_normalized.parquet"
S3_VAL_NORMALIZED = PROCESSED_DIR / "s3_val_normalized.parquet"
GROUND_TRUTH_TRAIN = PROCESSED_DIR / "ground_truth_train.parquet"
GROUND_TRUTH_VAL = PROCESSED_DIR / "ground_truth_val.parquet"
CANDIDATE_PAIRS_VAL = PROCESSED_DIR / "candidate_pairs_val.csv"

# Official Output Submission Paths
CANDIDATE_PAIRS_TEST = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_TEST = OUTPUT_DIR / "matching_results.tsv"

# Quality Gate Threshold
MIN_BLOCKING_RECALL = 0.90

# Blocking & Candidate Generation Parameters
# Tune these up if blocking recall is < 0.90
TFIDF_TOP_K = 30
MAX_CANDIDATES_PER_RECORD = 50
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MIN_DF = 1
TFIDF_MAX_FEATURES = 100000

# Training Parameters
VAL_SPLIT_RATIO = 0.20
RANDOM_STATE = 42
MATCH_THRESHOLD = 0.40  # Dynamically tuned to maximize Macro F_0.5

# Common Business Legal Suffixes & Abbreviations Map
BUSINESS_SYNONYMS = {
    "private limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt ltd.": "pvt ltd",
    "pvt.ltd": "pvt ltd",
    "pvtltd": "pvt ltd",
    "privatelimited": "pvt ltd",
    "limited": "ltd",
    "ltd.": "ltd",
    "corporation": "corp",
    "corp.": "corp",
    "incorporated": "inc",
    "inc.": "inc",
    "company": "co",
    "co.": "co",
    "enterprises": "ent",
    "enterprise": "ent",
    "services": "serv",
    "service": "serv",
    "international": "intl",
    "intl.": "intl",
    "technologies": "tech",
    "technology": "tech",
    "solutions": "sol",
    "solution": "sol",
    "llc": "llc",
    "l.l.c.": "llc",
    "llp": "llp",
    "l.l.p.": "llp",
}

# State and Common Location Abbreviations (US, India, global)
LOCATION_SYNONYMS = {
    # India
    "wb": "west bengal",
    "w.b.": "west bengal",
    "mh": "maharashtra",
    "dl": "delhi",
    "ncr": "national capital region",
    "ka": "karnataka",
    "tn": "tamil nadu",
    "up": "uttar pradesh",
    "u.p.": "uttar pradesh",
    "gj": "gujarat",
    "ts": "telangana",
    "ap": "andhra pradesh",
    "rj": "rajasthan",
    "mp": "madhya pradesh",
    "kl": "kerala",
    "pb": "punjab",
    "hr": "haryana",
    "br": "bihar",
    "or": "odisha",
    "as": "assam",
    "jh": "jharkhand",
    "ct": "chhattisgarh",
    "ut": "uttarakhand",
    "uk": "uttarakhand",
    "hp": "himachal pradesh",
    "ga": "goa",
    # US States
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming",
    # General address terms
    "rd": "road", "rd.": "road", "st": "street", "st.": "street",
    "ave": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "flr": "floor", "opp": "opposite", "nr": "near", "bldg": "building",
    "apt": "apartment", "sec": "sector", "hwy": "highway", "pkwy": "parkway"
}
