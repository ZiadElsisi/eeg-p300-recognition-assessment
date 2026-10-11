import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.synthetic_p300 import (
    generate_synthetic_p300
)

X, y, times = generate_synthetic_p300()

print("Shape:", X.shape)
print("Target trials:", (y == 1).sum())
print("NonTarget trials:", (y == 0).sum())
print("Epoch duration:", times[0], "to", times[-1])
print("Samples per epoch:", len(times))
