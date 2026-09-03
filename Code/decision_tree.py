import numpy as np


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  
        self.threshold = threshold      
        self.left = left                
        self.right = right              
        self.value = value             

    def is_leaf(self):
        return self.value is not None


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    p1 = np.mean(y)  
    p0 = 1 - p1
    return 1 - (p0 ** 2 + p1 ** 2)


class DecisionTreeScratch:
    def __init__(self, max_depth=10, min_samples_split=2, n_features_split=None,
                 seed=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_split = n_features_split  
        self.rng = np.random.default_rng(seed)
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        
        if (depth >= self.max_depth
                or n_samples < self.min_samples_split
                or gini_impurity(y) == 0.0):
            return Node(value=np.mean(y))

        
        if self.n_features_split is None:
            feature_indices = np.arange(n_features)
        else:
            feature_indices = self.rng.choice(
                n_features, size=self.n_features_split, replace=False
            )

        best_gain = -1.0
        best_feature, best_threshold = None, None

        parent_impurity = gini_impurity(y)

        for feat_idx in feature_indices:
            values = X[:, feat_idx]
            thresholds = np.unique(values)

            for t in thresholds:
                left_mask = values <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                
                n = n_samples
                weighted_impurity = (
                    len(y_left) / n * gini_impurity(y_left)
                    + len(y_right) / n * gini_impurity(y_right)
                )
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = t

        
        if best_feature is None:
            return Node(value=np.mean(y))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_idx=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child)

    def _predict_single(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict_proba(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_fake = rng.normal(size=(200, 2))
    y_fake = (X_fake[:, 0] + X_fake[:, 1] > 0).astype(int)

    tree = DecisionTreeScratch(max_depth=5, min_samples_split=5, seed=0)
    tree.fit(X_fake, y_fake)

    proba = tree.predict_proba(X_fake)
    preds = (proba >= 0.5).astype(int)
    accuracy = np.mean(preds == y_fake)
    print(f"Accuracy single tree: {accuracy:.3f}")