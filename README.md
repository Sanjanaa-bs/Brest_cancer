# 🧬 OncoPredict: Precision Medicine for Triple-Negative Breast Cancer

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Genomics-green)
![Status](https://img.shields.io/badge/Status-Research%20Active-brightgreen)

> **"Decoding the Genome to Predict Cancer Treatment Response."**

---

## 🎯 The Mission

Breast cancer is not a single disease. **Triple-Negative Breast Cancer (TNBC)** is its most aggressive form, lacking the three most common receptors (ER, PR, HER2) that targeted therapies rely on.

For TNBC patients, **Chemotherapy** is the primary weapon. But here's the problem:
*   Some patients achieve a **Pathologic Complete Response (pCR)** — the tumor disappears.
*   Others have **Residual Disease (RD)** — the tumor resists.

**OncoPredict** uses advanced Machine Learning to analyze gene expression data and predict *who* will respond to treatment before it even begins. This is the future of **Personalized Medicine**.

---

## 🔬 How It Works (The Research Pipeline)

We don't just dump data into a model. We built a robust, biological data refinery.

### 1. 📥 The Data Hub (Acquisition)
We aggregate massive genomic datasets from the **Gene Expression Omnibus (GEO)** and clinical data from **The Cancer Genome Atlas (TCGA)**.
*   *Datasets:* GSE25066, GSE20271, GSE20194, GSE32646
*   *Scope:* Hundreds of patient profiles with detailed molecular signatures.

### 2. 🧬 The Genomic Cleaner (Preprocessing)
Biological data is messy. Our pipeline:
*   **Harmonizes** gene names across different microarray platforms.
*   **Filters** specifically for TNBC samples.
*   **Corrects** technical batch effects using Z-score normalization.

### 3. proach The Feature Hunter (Hybrid Selection)
We find the needle in the haystack (the key genes) using a dual strategy:
*   **Biological Wisdom**: We prioritize genes known to be involved in DNA Repair (BRCA1/2) and Cell Cycle (MKI67).
*   **Data-Driven Discovery**: Using Recursive Feature Elimination (RFE) with Random Forests to find hidden patterns.

### 4. 🧠 The Prediction Engine
We train an ensemble of powerful classifiers to predict treatment outcome (pCR vs RD):
*   **XGBoost** & **Gradient Boosting** (The powerhouses)
*   **Random Forest** (The robust generalist)
*   **Support Vector Machines (SVM)** (The precise separator)

### 5. 🔍 The Explainer (SHAP)
A "black box" model is useless in medicine. We use **SHAP (SHapley Additive exPlanations)** to reveal *why* a prediction was made.
*   *Does high expression of Gene X mean better response?*
*   *Which molecular pathway is driving resistance?*

---

## 🚀 Quick Start (Reproduce the Science)

### Prerequisite
*   Python 3.9+
*   Virtual Environment (Highly Recommended)

### Step 1: Environment Setup 🛠️
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install scientific dependencies
pip install -r requirements.txt
```

### Step 2: Run the Pipeline 🏃‍♂️
Execute the scripts in order to replicate the full study.

```bash
# 1. Download Raw Data
python get-geo.py
python get-tcga.py

# 2. Clean & Normalize
python preprocess_pipeline.py
# Output: Cleaned gene expression matrix

# 3. Select Best Genes
python feature_engineering.py
# Output: Top 50, 100, 150 gene signatures

# 4. Train Models
python model_training.py
# Output: trained models, ROC curves, accuracy metrics

# 5. Explain Results
python model_interpretation.py
# Output: SHAP plots, gene importance rankings
```

---

## 📊 Key Results

The pipeline generates publication-quality visualization in the `results/` folder:
*   **ROC Curves**: Visualizing the trade-off between sensitivity and specificity.
*   **Gene Heatmaps**: Seeing the difference in expression between Responders (pCR) and Non-Responders (RD).
*   **SHAP Beeswarm Plots**: Identifying the exact genes driving the predictions.

---

## 🔮 Impact

By accurately predicting chemotherapy response, **OncoPredict** could help clinicians:
1.  **Spare non-responders** from the toxicity of ineffective chemotherapy.
2.  **Fast-track alternative therapies** (immunotherapy, PARP inhibitors) for high-risk patients.
3.  **Identify new drug targets** by understanding the biological mechanism of resistance.

---

## 🤝 Contributing to the Cure

We welcome contributions from bioinformaticians and data scientists!
1.  Fork the repo.
2.  Create a branch for your feature (e.g., `feature/DeepLearning-Model`).
3.  Submit a Pull Request.

---

**Made with ❤️ and 🧬 by Sanjana**
*Fighting Cancer with Code.*
