import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for i in range(self.n_iterations):
            z = X @ self.w + self.b
            p = sigmoid(z)

            error = p - y
            grad_w = (X.T @ error) / n_samples
            grad_b = np.sum(error) / n_samples

            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

            eps = 1e-15 
            p_clipped = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
            self.loss_history.append(loss)

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: loss = {loss:.4f}")

        return self

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_fake = rng.normal(size=(200, 2))
    y_fake = (X_fake[:, 0] + X_fake[:, 1] > 0).astype(int)

    model = LogisticRegressionScratch(learning_rate=0.5, n_iterations=500, verbose=True)
    model.fit(X_fake, y_fake)

    preds = model.predict(X_fake)
    accuracy = np.mean(preds == y_fake)
    print(f"\nTrain Accuracy: {accuracy:.3f}")
    print(f"Final weight: {model.w}, bias: {model.b:.3f}")