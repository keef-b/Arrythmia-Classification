# Arrhythmia Classification

Multi-class ECG beat classification using the MIT-BIH Arrhythmia Database.

This repository is for the five-class arrhythmia classifier, not the binary normal-vs-abnormal baseline.

## Classes

The dataset builder keeps five beat annotations and maps them to numeric labels:

| Label | Symbol | Beat type |
| --- | --- | --- |
| 0 | N | Normal beat |
| 1 | L | Left bundle branch block beat |
| 2 | R | Right bundle branch block beat |
| 3 | V | Premature ventricular contraction |
| 4 | A | Atrial premature beat |

## Project Structure

```text
.
├── data/
│   ├── raw/                 # Local MIT-BIH files; ignored by git
│   └── processed/           # Generated X.npy/y.npy files; ignored by git
├── models/
│   └── random_forest_multiclass.pkl
├── outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── scripts/
│   └── build_dataset.py     # Multi-class dataset builder
├── src/
│   ├── build_dataset.py
│   ├── train_random_forest.py
│   └── exploration/preprocessing helpers
└── requirements.txt
```

## Current Results

The committed run trains a random forest classifier on five beat classes.

```text
              precision    recall  f1-score   support

           N       0.99      1.00      0.99     17445
           L       0.99      0.99      0.99      1923
           R       1.00      0.99      0.99      1652
           V       0.97      0.96      0.97      1578
           A       0.97      0.84      0.90       613

    accuracy                           0.99     23211
   macro avg       0.98      0.96      0.97     23211
weighted avg       0.99      0.99      0.99     23211
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Download the MIT-BIH Arrhythmia Database locally and place it under `data/raw/`.

The raw data and generated NumPy arrays are intentionally not committed because they are large/generated artifacts.

## Build the Dataset

```powershell
python src\build_dataset.py
```

This creates:

```text
data/processed/X.npy
data/processed/y.npy
```

## Train the Model

```powershell
python src\train_random_forest.py
```

This writes:

```text
models/random_forest_multiclass.pkl
outputs/classification_report.txt
outputs/confusion_matrix.png
outputs/feature_importance.png
```

## Notes

- The binary normal-vs-abnormal project is kept separate from this repo.
- The current model is a strong baseline, but the atrial premature beat class has lower recall than the other classes and is the clearest next place to improve.
