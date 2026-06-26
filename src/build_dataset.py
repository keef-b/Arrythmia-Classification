import os
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW = 100

LABEL_MAP = {
    "N": 0,  # Normal beat
    "L": 1,  # Left bundle branch block beat
    "R": 2,  # Right bundle branch block beat
    "V": 3,  # Premature ventricular contraction
    "A": 4,  # Atrial premature beat
}

LABEL_NAMES = {
    0: "N",
    1: "L",
    2: "R",
    3: "V",
    4: "A",
}


def find_records(raw_data_dir: Path) -> list[Path]:
    records = []
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if filename.lower().endswith(".atr"):
                records.append(Path(root) / filename[:-4])
    return sorted(records)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records = find_records(RAW_DATA_DIR)

    print("=" * 60)
    print("MIT-BIH MULTI-CLASS DATASET BUILDER")
    print("=" * 60)
    print(f"Raw data folder: {RAW_DATA_DIR}")
    print(f"Found {len(records)} records")

    if not records:
        raise RuntimeError(f"No .atr files found under {RAW_DATA_DIR}")

    X = []
    y = []

    for idx, record_path in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] Processing {record_path.name}")

        try:
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(str(record_path), "atr")
        except Exception as exc:
            print(f"Skipping {record_path}: {exc}")
            continue

        signal = record.p_signal[:, 0]

        for sample, symbol in zip(annotation.sample, annotation.symbol):
            if symbol not in LABEL_MAP:
                continue

            start = sample - WINDOW
            end = sample + WINDOW
            if start < 0 or end >= len(signal):
                continue

            beat = signal[start:end]
            if len(beat) != 2 * WINDOW:
                continue

            beat_std = np.std(beat)
            if beat_std == 0:
                continue

            X.append((beat - np.mean(beat)) / beat_std)
            y.append(LABEL_MAP[symbol])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print("\nDataset summary")
    print("-" * 30)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    counts = Counter(y)
    for class_id in sorted(counts):
        print(f"{LABEL_NAMES[class_id]} ({class_id}): {counts[class_id]}")

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)

    print("\nSaved")
    print(PROCESSED_DIR / "X.npy")
    print(PROCESSED_DIR / "y.npy")


if __name__ == "__main__":
    main()
