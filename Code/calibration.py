import numpy as np
from logistic_regression import LogisticRegressionScratch


class PlattScaling:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr_model = LogisticRegressionScratch(
            learning_rate=learning_rate, n_iterations=n_iterations
        )

    def fit(self, proba_calib, y_calib):
        X = proba_calib.reshape(-1, 1)
        self.lr_model.fit(X, y_calib)
        return self

    def transform(self, proba):
        X = proba.reshape(-1, 1)
        return self.lr_model.predict_proba(X)


class IsotonicRegression:
    def __init__(self):
        self.x_thresholds = None   
        self.y_values = None       

    def fit(self, proba_calib, y_calib):
        order = np.argsort(proba_calib)
        x_sorted = proba_calib[order]
        y_sorted = y_calib[order].astype(float)

        
        blocks = [[y_sorted[i], 1, x_sorted[i], x_sorted[i]] for i in range(len(y_sorted))]

        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                mean_val, w1, xmin1, _ = blocks[i]
                mean_val2, w2, _, xmax2 = blocks[i + 1]
                new_weight = w1 + w2
                new_mean = (mean_val * w1 + mean_val2 * w2) / new_weight
                blocks[i] = [new_mean, new_weight, xmin1, xmax2]
                del blocks[i + 1]
               
                if i > 0:
                    i -= 1
            else:
                i += 1

        
        self.x_thresholds = np.array([b[3] for b in blocks])
        self.y_values = np.array([b[0] for b in blocks])
        return self

    def transform(self, proba):
       
        idx = np.searchsorted(self.x_thresholds, proba, side="left")
        idx = np.clip(idx, 0, len(self.y_values) - 1)
        return self.y_values[idx]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n)


    proba_raw = np.clip(y_true * 0.7 + 0.15 + rng.normal(0, 0.1, size=n), 0.01, 0.99)


    calib_idx = np.arange(0, n, 2)
    test_idx = np.arange(1, n, 2)

    proba_calib, y_calib = proba_raw[calib_idx], y_true[calib_idx]
    proba_test, y_test = proba_raw[test_idx], y_true[test_idx]

    platt = PlattScaling().fit(proba_calib, y_calib)
    proba_platt = platt.transform(proba_test)

    iso = IsotonicRegression().fit(proba_calib, y_calib)
    proba_iso = iso.transform(proba_test)

    print("Raw probabilities (first 5):", proba_test[:5].round(3))
    print("Post Platt scaling (first 5):", proba_platt[:5].round(3))
    print("Post isotonic regression (first 5):", proba_iso[:5].round(3))