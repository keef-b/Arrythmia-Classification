# Binary ECG Classification (Normal vs Abnormal)

This project contains the complete work completed so far for binary ECG beat classification using the MIT-BIH Arrhythmia Database.

## Project Goal

Train and evaluate a model that classifies each heartbeat into:
- `0`: Normal beat
- `1`: Abnormal beat

This folder is intentionally scoped to binary classification only.

## Project Structure

```
binary_normal_abnormal/
	data/
		raw/
			mit-bih-arrhythmia-database-1.0.0/   # Original MIT-BIH files (.hea/.atr/.xws, etc.)
		processed/
			X.npy                                # Feature matrix for training
			y.npy                                # Binary labels (0 normal, 1 abnormal)
	models/
		random_forest_binary.joblib            # Saved trained model
	outputs/
		confusion_matrix.png                   # Evaluation plot
	src/
		build_dataset.py
		ecg_preprocessing.py
		load_data.py
		plot_ecg.py
		train_random_forest.py
		verify_class_distribution.py
		beats_and_annotations.py
		explore_annotations.py
	README.md
```

## End-to-End Workflow

1. Place/keep the MIT-BIH raw dataset in `data/raw/mit-bih-arrhythmia-database-1.0.0/`.
2. Build the processed binary dataset:

	 ```powershell
	 cd src
	 python build_dataset.py
	 ```

3. Train and evaluate the random forest model:

	 ```powershell
	 python train_random_forest.py
	 ```

4. Review artifacts:
- Model file in `models/random_forest_binary.joblib`
- Confusion matrix image in `outputs/confusion_matrix.png`

## What Is Included So Far

- Data loading and preprocessing utilities
- Binary dataset generation (`X.npy`, `y.npy`)
- Random forest training and evaluation
- Class distribution and annotation exploration scripts

## Scope Boundary

This folder is the stable baseline for binary normal-vs-abnormal detection.

Any next phase (for example, multi-class arrhythmia case prediction) should be created as a separate sibling project folder to keep experiments and artifacts isolated.
