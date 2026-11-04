# NEU Steel Surface Defect Classification — Step 1

## Objective
Compare Gaussian Naive Bayes, Decision Tree, k-Nearest Neighbors, Support Vector Machine, and Random Forest on six industrial defect categories using features extracted from grayscale images. The end goal is a reproducible classification report, not bounding-box detection.

**Source:** [Kaggle — NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database).

Dataset: 1,800 grayscale images of 200×200 pixels, six balanced defect classes (300 each). The chosen Kaggle copy has `NEU-DET/train/images/<class>` and `NEU-DET/validation/images/<class>`; the latter will be **held out as final test data** and must not enter training, tuning, PCA, feature selection, or model selection. Only training images participate in 5-fold stratified cross-validation.

## Step 1 — Setup (Linux VS Code terminal)

```bash
cd /path/to/NEU_Steel_Defects_CPU
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/01_download_verify.py
```

If Kaggle login is required for downloading, visit https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database , download the ZIP, extract it into `data/raw/`, and run:

```bash
python scripts/01_download_verify.py --skip-download
```

This script scans the source folders, verifies image dimensions / grayscale format, reports class distributions and exact duplicates, and writes `outputs/step1_dataset_manifest.csv` and `outputs/step1_summary.json`.

Expected per class (for this Kaggle release): **240 train and 60 validation = 300 total**. Expected overall: **1,440 train and 360 validation = 1,800 images**.

### Important experiment constraints
- Reuse the publisher's split; the folder named `validation` is reserved exclusively for final testing in **our** protocol.
- Train on the `train` split; use stratified 5-fold CV on train only.
- Do not interpret this as accept/reject classification: no normal/defect-free category is provided.
- NEU images are available publicly, but Kaggle lists the license as **unknown**. Cite the dataset and verify rights before redistributing images.

### Original source citation
Song, K. & Yan, Y. (2013). *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects.* Applied Surface Science. https://doi.org/10.1016/j.apsusc.2013.05.009
