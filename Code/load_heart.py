import csv
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

DROP_COLUMNS = ["slope", "ca", "thal"]

FILES = [
    os.path.join(DATA_DIR, "processed.cleveland.data"),
    os.path.join(DATA_DIR, "processed.hungarian.data"),
    os.path.join(DATA_DIR, "processed.switzerland.data"),
    os.path.join(DATA_DIR, "processed.va.data"),
]

def load_heart_disease(files=FILES):
    rows = []
    for fpath in files:
        with open(fpath, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) != len(COLUMNS):
                    continue
                rows.append(line)

    data = np.array(rows, dtype=object)
    data[data == "?"] = np.nan
    data = data.astype(float)

    X_full = data[:, :-1]
    y_raw = data[:, -1]
    y = (y_raw > 0).astype(int)

    feature_names_full = COLUMNS[:-1]

    keep_idx = [i for i, name in enumerate(feature_names_full) if name not in DROP_COLUMNS]
    feature_names = [feature_names_full[i] for i in keep_idx]
    X = X_full[:, keep_idx]

    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            col[nan_mask] = median_val

    return X, y, feature_names


if __name__ == "__main__":
    X, y, feature_names = load_heart_disease()
    print("Feature names:", feature_names)
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Missing residual values:", np.isnan(X).sum(axis=0))
    print("class distribution:", np.bincount(y))