from pathlib import Path

def find_project_root(marker="Pipfile.lock"): ## The _name is a Python convention meaning private helper, only for use inside this file.
    for parent in Path(__file__).resolve().parents: ## parents are returned as addresses
        if (parent / marker).exists():
            print(parent)
            return parent
    raise RuntimeError(f"Could not find {marker}")

PROJECT_ROOT = find_project_root()
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
