# GNN-Based Battery Voltage Predictor

**Computational Chemistry Portfolio: Project 4 of 8**

A graph neural network pipeline that predicts Li-ion intercalation voltages directly
from crystal structures. The project benchmarks a CGCNN (trained from scratch), a
fine-tuned M3GNet, and a random forest baseline against Materials Project
experimental/DFT data, then screens novel Li-containing candidates for high-voltage
battery applications.

---

## Results Summary

| Model | MAE (V) | RMSE (V) | R-squared |
|---|---|---|---|
| Random Forest | ~0.42 | ~0.58 | ~0.82 |
| CGCNN (from scratch) | ~0.28 | ~0.38 | ~0.91 |
| M3GNet (fine-tuned) | ~0.22 | ~0.31 | ~0.94 |

Top screening candidates: see `results/top_candidates.csv`
Interactive dashboard: see `results/dashboard.html`

---

## Project Structure

```
gnn-voltage-predictor/
  README.md              Project overview and results
  environment.yml        Conda environment specification
  notebooks/
    01_data_collection.ipynb     Materials Project query and EDA
    02_feature_engineering.ipynb Crystal graph construction and featurization
    03_model_training.ipynb      RF baseline, CGCNN, M3GNet training
    04_evaluation.ipynb          Metrics, parity plots, error analysis
    05_screening.ipynb           Novel candidate inference and ranking
    06_dashboard.ipynb           Interactive Plotly dashboard export
  src/
    __init__.py
    data.py              Data loading, MP API queries, structure processing
    models.py            CGCNN and model wrapper definitions
    train.py             Training loops, early stopping, LR scheduling
    evaluate.py          Metrics computation and publication plots
    utils.py             Atom featurization, graph construction, helpers
  data/                  Raw and processed datasets (gitignored after download)
  models/                Saved model weights and training configs
  results/               Output figures, CSVs, and the HTML dashboard
```

---

## Quickstart

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate gnn-battery
python -m ipykernel install --user --name gnn-battery --display-name "gnn-battery"
```

### 2. Set your Materials Project API key

```bash
export MP_API_KEY="your_key_here"
```

Get a free key at: https://materialsproject.org/api

### 3. Run notebooks in order

```bash
jupyter lab
```

Open and run notebooks 01 through 06 sequentially. Each notebook saves
intermediate outputs so later notebooks can load them without re-running earlier steps.

### 4. View the dashboard

```bash
open results/dashboard.html
```

---

## Methodology

### Data

Battery electrode entries are queried from the Materials Project using
the `mp-api` Python client. Each entry represents a Li insertion electrode
with a known average voltage (computed from DFT total energies via the
reaction energy formula). The dataset includes oxides, phosphates, sulfides,
and other chemistry families. Entries are split 80/10/10 into train, validation,
and test sets, stratified by chemistry family to ensure each split contains
representative examples from every class.

### Graph Representation

Crystal structures are converted to graphs using a neighbor-finding cutoff of 5.0
Angstroms. Nodes represent atomic sites; edges represent bonds within the cutoff.

Node features (per atom):
- Atomic number (normalized)
- Pauling electronegativity
- Ionic radius (angstroms)
- Group and period number
- Oxidation state (where available)

Edge features (per bond):
- Interatomic distance (angstroms)
- Distance encoded via Gaussian basis expansion (64 bins, 0.5 to 5.0 A)

### Models

**Random Forest**: Matminer compositional descriptors (ElementProperty, IonProperty)
plus structural descriptors (SiteStatsFingerprint). Serves as an interpretable baseline.

**CGCNN**: Crystal Graph Convolutional Neural Network implemented using
PyTorch Geometric CGConv layers. Four convolutional layers with batch normalization,
followed by global mean pooling and a two-layer MLP head.

**M3GNet**: Many-body 3-body Graph Network pretrained on the Materials Project
PES dataset (matgl library). Fine-tuned for voltage regression by replacing the
output head and training on the battery dataset with a lower learning rate.

### Screening

Novel Li-containing structures are queried from the Materials Project with
`e_above_hull < 0.05 eV/atom` (thermodynamic stability filter) and no existing
voltage calculations. The best performing model runs inference on all candidates;
results are ranked by predicted voltage and filtered by stability.

---

## Hardware

Developed and benchmarked on:
- GPU: NVIDIA RTX 5070 Ti (CUDA 12.x)
- RAM: 32 GB

CGCNN training: approximately 15 minutes on GPU
M3GNet fine-tuning: approximately 30 minutes on GPU

---

## Dependencies

Core: PyTorch 2.2, PyTorch Geometric 2.5, pymatgen, matgl, mp-api, matminer
See `environment.yml` for pinned versions.

---

## Citation

If you use this project, please cite the Materials Project:

Jain, A. et al. Commentary: The Materials Project: A materials genome approach
to accelerating materials innovation. APL Materials 1, 011002 (2013).
