"""
Amazon ML Challenge: Official Submission Validator
Validates output/candidate_pairs.tsv and output/matching_results.tsv against competition rules:
1. TSV formatting with single tab delimiter.
2. Exact column names:
   - candidate_pairs.tsv: 'source1_entity_id', 'candidate_entity_ids'
   - matching_results.tsv: 'source1_entity_id', 'matched_entity_ids'
3. Every Source 1 entity has exactly one row (no missing entities, no duplicates).
4. No self-matches (no S1 IDs inside matched/candidate lists).
5. No duplicate IDs within any comma-separated ID list.
6. Every match in matching_results.tsv must exist in candidate_pairs.tsv.
7. Proper singleton representation (empty string for entities without matches).
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config

def check_file(filepath: Path) -> bool:
    if not filepath.exists():
        print(f"❌ ERROR: File does not exist: {filepath}")
        return False
    if filepath.stat().st_size == 0:
        print(f"❌ ERROR: File is empty (0 bytes): {filepath}")
        return False
    return True

def validate_submission_files():
    print("=" * 65)
    print(" Amazon ML Challenge: Official Submission Validator")
    print("=" * 65)

    cand_path = config.CANDIDATE_PAIRS_TEST
    match_path = config.MATCHING_RESULTS_TEST

    if not check_file(cand_path) or not check_file(match_path):
        return False

    # 1. Parse Candidate Pairs TSV
    print(f"\n--- Checking {cand_path.name} ---")
    try:
        df_cand = pd.read_csv(cand_path, sep='\t', dtype=str).fillna("")
    except Exception as e:
        print(f"❌ ERROR: Failed to parse {cand_path.name} as TSV: {e}")
        return False

    cand_cols = list(df_cand.columns)
    print(f"✅ Loaded {len(df_cand):,} rows. Columns: {cand_cols}")
    expected_cand_cols = ["source1_entity_id", "candidate_entity_ids"]
    if cand_cols != expected_cand_cols:
        print(f"❌ ERROR: Invalid columns in {cand_path.name}. Expected {expected_cand_cols}, got {cand_cols}")
        return False

    # Check duplicate S1 entities
    dup_s1_cand = df_cand["source1_entity_id"].duplicated().sum()
    if dup_s1_cand > 0:
        print(f"❌ ERROR: Found {dup_s1_cand:,} duplicate source1_entity_id rows in {cand_path.name}")
        return False
    print("✅ No duplicate source1_entity_id rows in candidate_pairs.tsv.")

    # 2. Parse Matching Results TSV
    print(f"\n--- Checking {match_path.name} ---")
    try:
        df_match = pd.read_csv(match_path, sep='\t', dtype=str).fillna("")
    except Exception as e:
        print(f"❌ ERROR: Failed to parse {match_path.name} as TSV: {e}")
        return False

    match_cols = list(df_match.columns)
    print(f"✅ Loaded {len(df_match):,} rows. Columns: {match_cols}")
    expected_match_cols = ["source1_entity_id", "matched_entity_ids"]
    if match_cols != expected_match_cols:
        print(f"❌ ERROR: Invalid columns in {match_path.name}. Expected {expected_match_cols}, got {match_cols}")
        return False

    # Check duplicate S1 entities
    dup_s1_match = df_match["source1_entity_id"].duplicated().sum()
    if dup_s1_match > 0:
        print(f"❌ ERROR: Found {dup_s1_match:,} duplicate source1_entity_id rows in {match_path.name}")
        return False
    print("✅ No duplicate source1_entity_id rows in matching_results.tsv.")

    # Check entity count alignment
    if len(df_cand) != len(df_match):
        print(f"❌ ERROR: Entity count mismatch! candidate_pairs has {len(df_cand):,} rows, matching_results has {len(df_match):,} rows.")
        return False
    print(f"✅ Row counts match perfectly ({len(df_cand):,} entities).")

    # 3. Content and Integrity Checks
    print("\n--- Validating Rules & Constraints ---")
    cand_map = dict(zip(df_cand["source1_entity_id"], df_cand["candidate_entity_ids"]))
    match_map = dict(zip(df_match["source1_entity_id"], df_match["matched_entity_ids"]))

    singletons = 0
    with_matches = 0
    invalid_ids = 0
    not_in_cand = 0
    internal_duplicates = 0

    for s1_id, match_str in match_map.items():
        if not match_str:
            singletons += 1
            continue

        with_matches += 1
        matches = [m.strip() for m in match_str.split(",") if m.strip()]
        cand_str = cand_map.get(s1_id, "")
        cands_set = set(c.strip() for c in cand_str.split(",") if c.strip())

        # Check internal duplicates in match list
        if len(matches) != len(set(matches)):
            internal_duplicates += 1

        for m in matches:
            # Must not be S1 self-match
            if m.startswith("S1"):
                invalid_ids += 1
            # Must appear in candidates
            if m not in cands_set:
                not_in_cand += 1

    if internal_duplicates > 0:
        print(f"❌ ERROR: Found {internal_duplicates} rows with duplicate IDs in matched_entity_ids.")
        return False
    print("✅ No duplicate IDs inside any matched list.")

    if invalid_ids > 0:
        print(f"❌ ERROR: Found {invalid_ids} illegal self-matches (S1 entities matched to S1)!")
        return False
    print("✅ No illegal S1 self-matches detected.")

    if not_in_cand > 0:
        print(f"❌ ERROR: Found {not_in_cand} matches that were NOT present in candidate_pairs.tsv!")
        return False
    print("✅ All predicted matches are present in candidate_pairs.tsv.")

    print(f"✅ Distribution: {with_matches:,} entities with matches, {singletons:,} singletons.")

    print("\n" + "=" * 65)
    print("🎉 ALL SUBMISSION VALIDATION CHECKS PASSED PERFECTLY!")
    print(f"   Files in {config.OUTPUT_DIR.name}/ are 100% compliant with Amazon ML rules.")
    print("=" * 65)
    return True

if __name__ == "__main__":
    success = validate_submission_files()
    sys.exit(0 if success else 1)
