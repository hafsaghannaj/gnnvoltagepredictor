# GNN-Based Battery Voltage Predictor

Computational Chemistry Portfolio: Project 4 of 8

Predicts Li-ion intercalation voltage from crystal structures using graph-based ML models, then screens novel Li-containing materials for high-voltage candidates.

## What's New (March 2026)

- Updated data ingestion to use `mp_api` insertion-electrode host structures (`src/data.py`), with backward-compatible structure keys for downstream graph building.
- Added and benchmarked a PyG `CrystalTransformer` model (`src/models.py`) alongside `CGCNN` and `Random Forest`.
- Added expanded evaluation outputs in `results/fig04_*` plus `results/benchmark_table.csv`.
- Added screening + ranking outputs for novel candidates in `results/screening_all.csv`, `results/top_candidates.csv`, and `results/top5_validation.csv`.
- Added interactive deliverables in `results/fig_screening_interactive.html`, `results/fig_voltage_distribution_interactive.html`, and `results/dashboard.html`.
- Added Captum-based graph explainability tools (`src/explain.py`) plus a runnable single-prediction attribution script (`scripts/explain_single_prediction.py`).
- Updated environment baseline to Python 3.11 and relaxed several package pins in `environment.yml`.

## Current Benchmark (Test Set)

Source: `results/benchmark_table.csv`

| Model | MAE (V) | RMSE (V) | R-squared |
|---|---:|---:|---:|
| CrystalTransformer | 0.4236 | 0.6161 | 0.6225 |
| Random Forest | 0.4475 | 0.6452 | 0.5860 |
| CGCNN | 0.4600 | 0.6822 | 0.5372 |

## Top 5 Screened Candidates

Source: `results/top_candidates.csv`

| Rank | Material ID | Formula | Family | Predicted Voltage (V) |
|---:|---|---|---|---:|
| 1 | mp-1020060 | LiB(S2O7)2 | sulfate | 4.423 |
| 2 | mp-9143 | LiPF6 | fluoride | 4.367 |
| 3 | mp-759185 | LiSb(PO3)4 | phosphate | 4.216 |
| 4 | mp-504353 | LiSb(PO3)4 | phosphate | 4.212 |
| 5 | mp-504207 | LiSb(PO3)4 | phosphate | 4.139 |

## Repository Layout

```text
gnn-voltage-predictor/
  README.md
  LICENSE
  environment.yml
  data/
  models/
    cgcnn_best.pt
    cgcnn_history.json
    rf_model.pkl
    rf_config.json
    transformer_best.pt
    transformer_history.json
  notebooks/
    01_data_collection.ipynb
    02_feature_engineering.ipynb
    03_model_training.ipynb
    04_evaluation.ipynb
    05_screening.ipynb
    06_dashboard.ipynb
  src/
    data.py
    models.py
    train.py
    evaluate.py
    utils.py
    explain.py
  scripts/
    explain_single_prediction.py
  results/
    benchmark_table.csv
    fig01_*.png
    fig04_*.png
    fig05_*.png
    screening_all.csv
    top_candidates.csv
    top5_validation.csv
    fig_screening_interactive.html
    fig_voltage_distribution_interactive.html
    dashboard.html
```

## Pipeline

1. Collect Li insertion electrode data from Materials Project.
2. Convert host structures to crystal graphs.
3. Train voltage predictors (RF, CGCNN, CrystalTransformer).
4. Evaluate on held-out test set with parity/error diagnostics.
5. Screen novel Li-containing structures and rank candidates.
6. Export static figures and interactive dashboard artifacts.

## Quick Start

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate gnn-battery
python -m ipykernel install --user --name gnn-battery --display-name "gnn-battery"
```

### 2. Set Materials Project API key

```bash
export MP_API_KEY="your_key_here"
```

### 3. Run notebooks in order

```bash
jupyter lab
```

Run notebooks `01` through `06` sequentially.

### 4. Open interactive outputs

```bash
open results/dashboard.html
open results/fig_screening_interactive.html
open results/fig_voltage_distribution_interactive.html
```

## Explainability (Single Prediction)

Use Captum Integrated Gradients (`ig`) or SHAP-like GradientShap (`gradient_shap`)
to attribute one graph prediction and identify the most influential atoms.

```bash
python scripts/explain_single_prediction.py \
  --model transformer \
  --method ig \
  --graph-index 0 \
  --top-k 10
```

Outputs are saved to `results/`, including:

- top-atom bar plot (`*_top_atoms.png`)
- 3D atom influence map (`*_structure_3d.png`, when structure metadata is available)
- full attribution payload (`*.json`)

## Notes

- The repository includes model artifacts and generated results so outputs are immediately inspectable.
- Screening predictions are model-based estimates; prioritize DFT/experimental validation before synthesis decisions.

## Citation

If you use this project, please cite the Materials Project:

Jain, A. et al. (2013). The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1, 011002.
