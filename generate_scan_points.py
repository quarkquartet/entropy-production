"""Generate the 8,000 parameter points used for the paper's Z2 scan."""

import csv
from pathlib import Path

import numpy as np


output_dir = Path(__file__).resolve().parent / "output"


def random_rectangle_points(count=8000, seed=20261025):
    rng = np.random.default_rng(seed)
    return rng.uniform([62.5, 0.55], [160.0, 1.4], size=(count, 2))


def main():
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "uniform_rectangle_points.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_index", "mass", "lambda_hs"])
        for index, (mass, lambda_hs) in enumerate(random_rectangle_points()):
            # Match the parameter precision passed to the production calculation.
            writer.writerow([index, f"{mass:.12g}", f"{lambda_hs:.12g}"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
