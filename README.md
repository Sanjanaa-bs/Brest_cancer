# Breast Cancer Treatment Response Prediction (pCR)

## Overview

This project implements a comprehensive Machine Learning pipeline to predict **Pathologic Complete Response (pCR)** in breast cancer patients, specifically focusing on **Triple-Negative Breast Cancer (TNBC)**. It utilizes gene expression data from multiple GEO datasets and TCGA clinical data to identify molecular signatures associated with treatment response.

The pipeline includes data acquisition, rigorous preprocessing, feature engineering with biological prior knowledge, machine learning model training (Random Forest, XGBoost, etc.), and model interpretation using SHAP values.

## Table of Contents

- [Breast Cancer Treatment Response Prediction (pCR)](#breast-cancer-treatment-response-prediction-pcr)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Data Sources](#data-sources)
  - [Usage Pipeline](#usage-pipeline)
    - [1. Data Acquisition](#1-data-acquisition)
    - [2. Preprocessing](#2-preprocessing)
    - [3. Feature Engineering](#3-feature-engineering)
    - [4. Model Training](#4-model-training)
    - [5. Model Interpretation](#5-model-interpretation)
  - [Key Features](#key-features)
  - [Results](#results)

## Project Structure

```
Brest_cancer_ml/
├── data/                   # Data directory (created during execution)
│   ├── raw/                # Raw downloads from GEO/TCGA
│   └── processed/          # Cleaned and harmonized data
├── models/                 # Saved model artifacts
├── notebooks/              # Jupyter notebooks for exploration
├── results/                # Analysis outputs, plots, and reports
├── src/                    # Source code modules
├── get-geo.py             # Script to download GEO datasets
├── get-tcga.py            # Script to download TCGA data
├── preprocess_pipeline.py  # Data cleaning and harmonization (Step 2)
├── feature_engineering.py  # Feature selection & RFE (Step 3)
├── model_training.py       # ML Training pipeline (Step 4)
├── model_interpretation.py # SHAP analysis & Viz (Step 5)
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project folder.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Sources

The project utilizes the following datasets:

- **GEO Datasets**: GSE25066, GSE20271, GSE20194, GSE32646 (Gene expression profiles)
- **TCGA-BRCA**: Clinical data from The Cancer Genome Atlas Breast Invasive Carcinoma project.

## Usage Pipeline

Run the scripts in the following order to reproduce the analysis.

### 1. Data Acquisition

Download the necessary gene expression and clinical data.

```bash
python get-geo.py
python get-tcga.py
```

- Use `verify_downloads.py` to ensure data integrity.
- Use `fix_GSE32646_download.py` if you encounter issues with that specific dataset.

### 2. Preprocessing

Clean, harmonize, and batch-correct the data. This pipeline handles TNBC filtering, gene name standardization, and Z-score normalization.

```bash
python preprocess_pipeline.py
```

- **Output**: `data/processed/combined/geo_expression_combined.csv`

### 3. Feature Engineering

Select the most predictive genes using a combination of:

- **Prior Knowledge**: Known DNA repair and cell cycle markers.
- **Differential Expression**: Genes significantly different between pCR and RD groups.
- **RFE (Recursive Feature Elimination)**: Random Forest-based selection.

```bash
python feature_engineering.py
```

- **Output**: Top 50, 100, and 150 gene signatures.

### 4. Model Training

Train and evaluate multiple machine learning models (RF, XGBoost, SVM, etc.) using 5-fold cross-validation and SMOTE for class imbalance.

```bash
python model_training.py
```

- **Output**: Performance metrics, ROC curves, and saved models in `results/models_topX`.

### 5. Model Interpretation

Analyze model decisions using SHAP (SHapley Additive exPlanations) to identify potential biomarkers.

```bash
python model_interpretation.py
```

- **Output**: SHAP summary plots, feature importance heatmaps, and consensus gene rankings.

## Key Features

- **Robust Preprocessing**: Handles missing data, batch effects, and cross-platform gene mapping.
- **Hybrid Feature Selection**: Combines biological prior knowledge with data-driven selection.
- **Ensemble Modeling**: Evaluates multiple algorithms to find the best predictor.
- **Explainable AI**: detailed SHAP analysis to provide biological context to predictions.

## Results

The pipeline generates detailed reports and visualizations in the `results/` directory, including:

- **Performance Comparison**: Accuracy, AUROC, Precision, Recall, F1-scores.
- **Gene Heatmaps**: Expression patterns of top predictive genes.
- **Feature Importance**: Ranked lists of genes contributing to treatment response prediction.
