import numpy as np
from pinaka.core.tensor import Tensor
from pinaka.models.linear import LinearRegression


def test_linear_regression_fit_predict():
    # Dataset: y = 2*x1 + 3*x2 + 1
    X = Tensor([[1, 2], [2, 0], [3, 1], [4, 3]])
    y = Tensor(
        [2 * 1 + 3 * 2 + 1, 2 * 2 + 3 * 0 + 1, 2 * 3 + 3 * 1 + 1, 2 * 4 + 3 * 3 + 1]
    )

    model = LinearRegression(lr=0.01, n_iters=5000)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.allclose(preds, y)

    assert np.allclose(model.weights.data, Tensor([2.0, 3.0]).data, atol=1e-2)

    assert np.allclose(model.bias.data, Tensor(1.0).data, atol=1e-2)
