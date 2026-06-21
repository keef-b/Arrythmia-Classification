import os
from collections import Counter

import numpy as np
import wfdb

# ============================================================
# CONFIGURATION
# ============================================================

WINDOW = 100  # samples before and after annotation

LABEL_MAP = {
    "N": 0,  # Normal
    "L": 1,  # Left Bundle Branch Block
    "R": 2,  # Right Bundle Branch Block
    "V": 3,  # PVC
    "A": 4,  # Atrial Premature Beat
}

LABEL_NAMES = {
    0: "N",
    1: "L",
    2: "R",
    3: "V",
    4: "A",
}

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(__file__)

BASE_RAW = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "data",
        "raw"
    )
)

OUTPUT_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "data",
        "processed"
    )
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FIND ALL RECORDS
# ============================================================

records = []

for root, dirs, files in os.walk(BASE_RAW):
    for f in files:

        if f.lower().endswith(".atr"):

            record_name = f[:-4]

            records.append(
                os.path.join(root, record_name)
            )

records.sort()

print("=" * 60)
print("MIT-BIH ARRHYTHMIA DATASET BUILDER")
print("=" * 60)

print(f"\nRaw data folder:")
print(BASE_RAW)

print(f"\nFound {len(records)} records")

if len(records) == 0:
    raise RuntimeError(
        f"No .atr files found under:\n{BASE_RAW}"
    )

print("\nFirst few records:")

for r in records[:5]:
    print(r)

# ============================================================
# BUILD DATASET
# ============================================================

X = []
y = []

for i, record_path in enumerate(records, start=1):

    print(f"\n[{i}/{len(records)}] Processing {os.path.basename(record_path)}")

    try:

        record = wfdb.rdrecord(record_path)

        annotation = wfdb.rdann(
            record_path,
            "atr"
        )

        signal = record.p_signal[:, 0]

        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol
        ):

            if symbol not in LABEL_MAP:
                continue

            start = sample - WINDOW
            end = sample + WINDOW

            # Skip beats too close to signal edges
            if start < 0:
                continue

            if end >= len(signal):
                continue

            beat = signal[start:end]

            # Ensure fixed length
            if len(beat) != 2 * WINDOW:
                continue

            # Normalize beat
            beat_std = np.std(beat)

            if beat_std == 0:
                continue

            beat = (
                beat - np.mean(beat)
            ) / beat_std

            X.append(beat)
            y.append(LABEL_MAP[symbol])

    except Exception as e:

        print(f"ERROR processing {record_path}")
        print(e)

# ============================================================
# CONVERT TO NUMPY
# ============================================================

print("\nConverting to numpy arrays...")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

counts = Counter(y)

print("\nClass Distribution")

for class_id in sorted(counts.keys()):

    print(
        f"{LABEL_NAMES[class_id]:>2} "
        f"({class_id}) : "
        f"{counts[class_id]}"
    )

# ============================================================
# SAVE DATASET
# ============================================================

x_path = os.path.join(
    OUTPUT_DIR,
    "X.npy"
)

y_path = os.path.join(
    OUTPUT_DIR,
    "y.npy"
)

np.save(x_path, X)
np.save(y_path, y)

print("\nSaved Files")
print("-" * 30)
print(x_path)
print(y_path)

print("\nDataset build complete.")