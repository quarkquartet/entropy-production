#!/bin/bash
# Run from this repository: bash run_scan.sh [number_of_parallel_points].
# Generate the paper's random points and calculate output/vw_scan.csv.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python generate_scan_points.py
python scan.py --workers "${1:-1}"
