import numpy as np


def train_calibration_test_split(X, y, train_frac=0.6, calib_frac=0.2,
                                    seed=42):
    assert train_frac + calib_frac < 1.0, "train_frac + calib_frac must be < 1"

    rng = np.random.default_rng(seed)

    X_train_parts, y_train_parts = [], []
    X_calib_parts, y_calib_parts = [], []
    X_test_parts, y_test_parts = [], []

    classes = np.unique(y)

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        n_c = len(idx_c)
        n_train = int(n_c * train_frac)
        n_calib = int(n_c * calib_frac)

        train_idx = idx_c[:n_train]
        calib_idx = idx_c[n_train:n_train + n_calib]
        test_idx = idx_c[n_train + n_calib:]

        X_train_parts.append(X[train_idx])
        y_train_parts.append(y[train_idx])

        X_calib_parts.append(X[calib_idx])
        y_calib_parts.append(y[calib_idx])

        X_test_parts.append(X[test_idx])
        y_test_parts.append(y[test_idx])

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    X_calib = np.concatenate(X_calib_parts, axis=0)
    y_calib = np.concatenate(y_calib_parts, axis=0)
    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    def shuffle_together(X_, y_):
        perm = rng.permutation(len(y_))
        return X_[perm], y_[perm]

    X_train, y_train = shuffle_together(X_train, y_train)
    X_calib, y_calib = shuffle_together(X_calib, y_calib)
    X_test, y_test = shuffle_together(X_test, y_test)

    return X_train, y_train, X_calib, y_calib, X_test, y_test


if __name__ == "__main__":

    X_fake = np.arange(20).reshape(10, 2)
    y_fake = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X_tr, y_tr, X_cal, y_cal, X_te, y_te = train_calibration_test_split(
        X_fake, y_fake, train_frac=0.6, calib_frac=0.2, seed=0
    )
    print("Train:", X_tr.shape, "class:", np.bincount(y_tr))
    print("Calib:", X_cal.shape, "class:", np.bincount(y_cal))
    print("Test:", X_te.shape, "class:", np.bincount(y_te))

def standardize(X_train, X_calib, X_test):

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_calib_scaled = (X_calib - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_calib_scaled, X_test_scaled, mean, std    