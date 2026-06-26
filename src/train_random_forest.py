from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

try:
    import seaborn as sns

    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

CLASS_NAMES = ["N", "L", "R", "V", "A"]
CLASS_DESCRIPTIONS = {
    0: "Normal beat",
    1: "Left bundle branch block beat",
    2: "Right bundle branch block beat",
    3: "Premature ventricular contraction",
    4: "Atrial premature beat",
}


def normalize_beats(X: np.ndarray) -> np.ndarray:
    return (X - np.mean(X, axis=1, keepdims=True)) / (
        np.std(X, axis=1, keepdims=True) + 1e-8
    )


def save_confusion_matrix(cm: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    if SNS_AVAILABLE:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={"label": "Count"},
            ax=ax,
        )
    else:
        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_title("Multi-Class ECG Beat Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model: RandomForestClassifier, n_features: int) -> None:
    importances = model.feature_importances_
    feature_idx = np.arange(n_features)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(feature_idx, importances)
    ax.set_title("Random Forest Feature Importance by Beat Sample")
    ax.set_xlabel("Sample index in beat window")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 60)
    print("MULTI-CLASS ECG ARRHYTHMIA CLASSIFICATION")
    print("=" * 60)

    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")

    print(f"Samples: {X.shape[0]:,}")
    print(f"Features per sample: {X.shape[1]}")

    print("\nClass distribution")
    print("-" * 30)
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y) * 100
        description = CLASS_DESCRIPTIONS.get(int(label), "Unknown")
        print(f"{CLASS_NAMES[int(label)]} ({int(label)}): {count:>7,} ({pct:5.1f}%) - {description}")

    X = normalize_beats(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    print("\nTraining random forest...")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
    test_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")

    print("\nPerformance")
    print("-" * 30)
    print(f"Train accuracy:       {train_acc:.4f}")
    print(f"Test accuracy:        {test_acc:.4f}")
    print(f"Balanced accuracy:    {test_balanced_acc:.4f}")
    print(f"Macro F1:             {test_macro_f1:.4f}")
    print(f"Weighted F1:          {test_weighted_f1:.4f}")
    print(f"Accuracy gap:         {train_acc - test_acc:+.4f}")

    report = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES)
    print("\nClassification report")
    print("-" * 30)
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_test_pred)
    save_confusion_matrix(cm)
    save_feature_importance(model, X.shape[1])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "random_forest_multiclass.pkl"
    joblib.dump(model, model_path)

    print("\nSaved artifacts")
    print("-" * 30)
    print(model_path)
    print(OUTPUT_DIR / "classification_report.txt")
    print(OUTPUT_DIR / "confusion_matrix.png")
    print(OUTPUT_DIR / "feature_importance.png")


if __name__ == "__main__":
    main()
