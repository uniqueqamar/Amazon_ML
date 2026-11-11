"""
Helper to create the official Amazon ML Challenge submission zip archive:
Structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── utils/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import sys
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def create_submission_zip(team_name: str = "team_submission"):
    output_zip = PROJECT_ROOT / f"{team_name}.zip"
    print(f"Creating submission zip: {output_zip.name}...")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Output files
        for f in (PROJECT_ROOT / "output").glob("*.tsv"):
            zipf.write(f, arcname=f"output/{f.name}")
            print(f"  + output/{f.name}")

        # 2. Source code
        for py_file in (PROJECT_ROOT / "src").glob("*.py"):
            zipf.write(py_file, arcname=f"code/business_entity_resolution/src/{py_file.name}")
            print(f"  + code/business_entity_resolution/src/{py_file.name}")

        for util_file in (PROJECT_ROOT / "utils").glob("*.py"):
            zipf.write(util_file, arcname=f"code/business_entity_resolution/utils/{util_file.name}")
            print(f"  + code/business_entity_resolution/utils/{util_file.name}")

        if (PROJECT_ROOT / "README.md").exists():
            zipf.write(PROJECT_ROOT / "README.md", arcname="code/business_entity_resolution/README.md")
        if (PROJECT_ROOT / "requirements.txt").exists():
            zipf.write(PROJECT_ROOT / "requirements.txt", arcname="code/business_entity_resolution/requirements.txt")
        if (PROJECT_ROOT / "run_pipeline.sh").exists():
            zipf.write(PROJECT_ROOT / "run_pipeline.sh", arcname="code/business_entity_resolution/run_pipeline.sh")

        # 3. Documentation template
        doc_file = PROJECT_ROOT / "Documentation_template.md"
        if doc_file.exists():
            zipf.write(doc_file, arcname="Documentation_template.md")
            print("  + Documentation_template.md")

    print(f"\n✅ Packaged submission archive: {output_zip}")
    print(f"   Archive size: {output_zip.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    t_name = sys.argv[1] if len(sys.argv) > 1 else "team_submission"
    create_submission_zip(t_name)
