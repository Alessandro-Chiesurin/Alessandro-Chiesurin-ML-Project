import numpy as np
from load_heart import load_heart_disease
from load_student import load_student_dropout
from utils import train_calibration_test_split, standardize
from logistic_regression import LogisticRegressionScratch
from random_forest import RandomForestScratch
from metrics import evaluate_model, plot_reliability_diagram
from calibration import PlattScaling, IsotonicRegression


def prepare_dataset(load_fn, name, seed=42):
    X, y, feature_names = load_fn()
    X_train, y_train, X_calib, y_calib, X_test, y_test = train_calibration_test_split(
        X, y, train_frac=0.6, calib_frac=0.2, seed=seed
    )
    X_train, X_calib, X_test, mean, std = standardize(X_train, X_calib, X_test)

    print(f"\n=== {name} ===")
    print("Train:", X_train.shape, "class:", np.bincount(y_train))
    print("Calib:", X_calib.shape, "class:", np.bincount(y_calib))
    print("Test: ", X_test.shape, "class:", np.bincount(y_test))

    return X_train, y_train, X_calib, y_calib, X_test, y_test, feature_names


def run_full_pipeline(X_train, y_train, X_calib, y_calib, X_test, y_test,
                        dataset_name, seed=42):
    models = {
        "logistic_regression": LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000),
        "random_forest": RandomForestScratch(n_trees=50, max_depth=10,
                                              min_samples_split=5, seed=seed),
    }

    all_results = {}

    for model_name, model in models.items():
        print(f"\n--- {dataset_name} | {model_name} ---")

        model.fit(X_train, y_train)

        proba_calib_raw = model.predict_proba(X_calib)
        proba_test_raw = model.predict_proba(X_test)

        metrics_raw = evaluate_model(y_test, proba_test_raw)
        print(f"Baseline (no calibration):  {metrics_raw}")

        platt = PlattScaling().fit(proba_calib_raw, y_calib)
        proba_test_platt = platt.transform(proba_test_raw)
        metrics_platt = evaluate_model(y_test, proba_test_platt)
        print(f"Post Platt scaling:          {metrics_platt}")

        iso = IsotonicRegression().fit(proba_calib_raw, y_calib)
        proba_test_iso = iso.transform(proba_test_raw)
        metrics_iso = evaluate_model(y_test, proba_test_iso)
        print(f"Post isotonic regression:    {metrics_iso}")

        all_results[model_name] = {
            "model": model,
            "proba_raw": proba_test_raw,
            "proba_platt": proba_test_platt,
            "proba_iso": proba_test_iso,
            "metrics_raw": metrics_raw,
            "metrics_platt": metrics_platt,
            "metrics_iso": metrics_iso,
        }

    return all_results


if __name__ == "__main__":
    # --- Heart Disease ---
    (X_train_h, y_train_h, X_calib_h, y_calib_h,
     X_test_h, y_test_h, features_h) = prepare_dataset(load_heart_disease, "Heart Disease")

    results_heart = run_full_pipeline(
        X_train_h, y_train_h, X_calib_h, y_calib_h, X_test_h, y_test_h, "Heart Disease"
    )

    # --- Student Dropout ---
    (X_train_s, y_train_s, X_calib_s, y_calib_s,
     X_test_s, y_test_s, features_s) = prepare_dataset(load_student_dropout, "Student Dropout")

    results_student = run_full_pipeline(
        X_train_s, y_train_s, X_calib_s, y_calib_s, X_test_s, y_test_s, "Student Dropout"
    )

    # --- Reliability diagrams ---
    datasets_results = [
        ("Heart Disease", results_heart, y_test_h),
        ("Student Dropout", results_student, y_test_s),
    ]

    for dataset_name, results, y_test in datasets_results:
        for model_name, res in results.items():
            proba_dict = {
                "Raw": res["proba_raw"],
                "Platt": res["proba_platt"],
                "Isotonic": res["proba_iso"],
            }
            title = f"{dataset_name} - {model_name}"
            save_path = f"reliability_{dataset_name.replace(' ', '_')}_{model_name}.png"
            plot_reliability_diagram(y_test, proba_dict, n_bins=10,
                                       title=title, save_path=save_path)