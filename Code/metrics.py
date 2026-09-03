import numpy as np
import matplotlib.pyplot as plt


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def log_loss_score(y_true, y_proba, eps=1e-15):
    p = np.clip(y_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def brier_score(y_true, y_proba):
    return np.mean((y_proba - y_true) ** 2)


def reliability_diagram_data(y_true, y_proba, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_freq = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)

        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            observed_freq[i] = y_true[mask].mean()

    return bin_centers, observed_freq, bin_counts


def evaluate_model(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss_score(y_true, y_proba),
        "brier_score": brier_score(y_true, y_proba),
    }


def plot_reliability_diagram(y_true, proba_dict, n_bins=10, title="Reliability Diagram", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    colors = ["tab:blue", "tab:orange", "tab:green"]

    
    def counts_to_sizes(counts, min_size=20, max_size=300):
        counts = np.asarray(counts, dtype=float)
        if counts.max() == counts.min():
            return np.full_like(counts, (min_size + max_size) / 2)
        norm = (counts - counts.min()) / (counts.max() - counts.min())
        return min_size + norm * (max_size - min_size)

    for (label, proba), color in zip(proba_dict.items(), colors):
        centers, freq, counts = reliability_diagram_data(y_true, proba, n_bins=n_bins)
        valid = ~np.isnan(freq)

        
        ax.plot(centers[valid], freq[valid], color=color, linewidth=1, alpha=0.6)

        
        sizes = counts_to_sizes(counts[valid])
        ax.scatter(centers[valid], freq[valid], s=sizes, color=color,
                   label=label, edgecolors="black", linewidth=0.5, zorder=3)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)