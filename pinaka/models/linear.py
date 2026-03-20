from pinaka.core.tensor import Tensor


class LinearRegression:
    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def predict(self, X: Tensor) -> Tensor:
        return X @ self.weights + self.bias

    def fit(self, X: Tensor, y: Tensor):
        n_samples, n_features = X.shape
        self.weights = Tensor([0.0] * n_features)
        self.bias = Tensor(0)

        for i in range(self.n_iters):
            y_pred = self.predict(X)

            dw = ((y_pred - y) @ X) * (1 / n_samples)
            db = (y_pred - y).sum() * (1 / n_samples)

            self.weights = self.weights - dw * self.lr
            self.bias = self.bias - db * self.lr
