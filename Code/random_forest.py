import numpy as np
from decision_tree import DecisionTreeScratch


class RandomForestScratch:
    def __init__(self, n_trees=50, max_depth=10, min_samples_split=2,
                 n_features_split="sqrt", seed=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_split_mode = n_features_split
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def _resolve_n_features(self, n_features):
        if self.n_features_split_mode == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.n_features_split_mode is None:
            return n_features
        else:
            return self.n_features_split_mode

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_feat_split = self._resolve_n_features(n_features)

        self.trees = []
        for i in range(self.n_trees):
            
            bootstrap_idx = self.rng.integers(0, n_samples, size=n_samples)
            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]

            
            tree_seed = self.rng.integers(0, 1_000_000)

            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features_split=n_feat_split,
                seed=tree_seed,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])
        return all_proba.mean(axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_fake = rng.normal(size=(200, 2))
    y_fake = (X_fake[:, 0] + X_fake[:, 1] > 0).astype(int)

    forest = RandomForestScratch(n_trees=20, max_depth=5, min_samples_split=5, seed=0)
    forest.fit(X_fake, y_fake)

    proba = forest.predict_proba(X_fake)
    preds = (proba >= 0.5).astype(int)
    accuracy = np.mean(preds == y_fake)
    print(f"Random forest accuracy: {accuracy:.3f}")
    print(f"Probability example (first 5): {proba[:5].round(3)}")