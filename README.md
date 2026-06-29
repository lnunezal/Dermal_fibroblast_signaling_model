# Dermal Fibroblast Signaling Model

Code and data for the manuscript *"Coupling fibroblast mechanotransduction signaling to tissue growth in a multiscale model of skin expansion"* (Nunez-Alvarez et al.).

## Structure

```
code/
├── network/             ODE model + Netflux-generated source files
├── calibration/         Two-stage Blackbox calibration of the signaling network
├── analysis/            Scripts/notebooks that generate the main-text figures
└── tissue_coupling/     Tissue-level growth coupling (G1 and G2 directions)

data/
├── calibration/         Curated stimulation, knockdown, and network tables
└── tissue_expansion/    Porcine tissue-expansion growth data

results/
├── calibration_logs/    Blackbox optimizer loss trajectories
├── KD_matrices/         Per-tension knockdown matrices
└── metrics/             Aggregated ΔActivity metrics
```

## Running the code

Notebooks expect to find the input data on the path; the simplest setup is to place the `data/` files next to the notebook you are running, or adjust the data paths in the first cell.

Order of execution to reproduce the paper from scratch:

1. `code/calibration/BB_run_DOF_cluster.py` — Stage 2 calibration of the cluster weights.
2. `code/analysis/CM_metrics_text.ipynb` — generates the four KD matrices and the per-phenotype metrics.
3. `code/analysis/CM_figure3_signalingcontexts.ipynb` — Fig. 4 and Fig. S2.
4. `code/analysis/figure5_tgfb_tension_with_zoom_insets.py` — Fig. 5.
5. `code/tissue_coupling/coupling_G1_optimizer.py` and `coupling_G2_optimizer_BB_angles_perTheta.py` — tissue-coupling calibration.
6. `code/tissue_coupling/CM_model_predictions_perTheta.ipynb` and `model_predictions_perTheta_wG2.ipynb` — Fig. 7 (G1) and Fig. S7 (G2).

## Dependencies

Python 3.10+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `jupyter`, and the `blackbox` global optimizer (Knysh & Korkolis 2016).
