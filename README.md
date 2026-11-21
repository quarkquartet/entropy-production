# Determining Bubble Wall Velocity from Entropy Production

This repository contains the entropy-production calculations used to determine bubble wall velocities in our paper (arXiv:26xx.xxxxx).
If you use this code or build on these calculations, please cite the paper and this repository.

The implementation supports both general and template equations of state.
For a given trial wall velocity, it computes the hydrodynamic profiles and microscopic entropy production, then searches for roots of the entropy residual to identify steady deflagration, hybrid, and detonation solutions.
The WallGo comparison results presented in the paper are calculated separately.

The example scripts illustrate the calculations presented in the paper.
`benchmarks.py` evaluates BP1 and BP2, the benchmark points discussed in Fig. 3 and Table 2.
`generate_scan_points.py` generates the random parameter points, and `scan.py` calculates their transition properties and wall velocities, writing the results to `output/vw_scan.csv`.
`run_scan.sh` shows the execution sequence for the scan.
You can adapt the scan setup to your local computer or cluster environment.
