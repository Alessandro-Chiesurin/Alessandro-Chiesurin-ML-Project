import csv
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

FILE_PATH = os.path.join(DATA_DIR, "student.dropout.csv")

TARGET_MAP = {
    "Dropout": 1,
    "Enrolled": 0,
    "Graduate": 0,
}


def load_student_dropout(fpath=FILE_PATH):
    with open(fpath, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader) 
        rows = [line for line in reader if len(line) == len(header)]

    header = [h.strip() for h in header]
    feature_names = header[:-1]

    data_str = np.array(rows, dtype=object)

    X = data_str[:, :-1].astype(float)
    y_text = data_str[:, -1]

    y = np.array([TARGET_MAP[label] for label in y_text], dtype=int)

    return X, y, feature_names


if __name__ == "__main__":
    X, y, feature_names = load_student_dropout()
    print("Feature names:", feature_names)
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Missing values by column:", np.isnan(X).sum(axis=0))
    print("Class Distribution (0=non-dropout, 1=dropout):", np.bincount(y))