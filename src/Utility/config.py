import hashlib
from pathlib import Path

def find_project_root(marker="Pipfile.lock"): ## The _name is a Python convention meaning private helper, only for use inside this file.
    for parent in Path(__file__).resolve().parents: ## parents are returned as addresses
        if (parent / marker).exists():
            print(parent)
            return parent
    raise RuntimeError(f"Could not find {marker}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

PROJECT_ROOT = find_project_root()
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
