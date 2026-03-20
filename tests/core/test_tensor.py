import pytest
import numpy as np
from pinaka.core.tensor import Tensor


@pytest.fixture
def t():
    return Tensor([1, 2, 3, 4, 5])


# Tensor Initialization


def test_tensor_init_list(t):
    assert isinstance(t.data, np.ndarray)
    assert t.shape == (5,)
    assert t.dtype == np.float64


def test_tensor_init_numpy():
    arr = np.array([[1, 2], [3, 4]])
    t = Tensor(arr)
    assert t.shape == (2, 2)
    assert t.dtype == np.float64


# Arithmetic operations


def test_tensor_add_scalar(t):
    r = t + 5
    assert np.all(r.to_numpy() == np.array([6, 7, 8, 9, 10]))


def test_tensor_add_tensor():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    r = t1 + t2
    assert np.all(r.to_numpy() == np.array([5, 7, 9]))


def test_tesnsor_mul_scalar(t):
    r = t * 10
    assert np.all(r.to_numpy() == np.array([10, 20, 30, 40, 50]))


def test_tensor_sub_scalar(t):
    r = t - 1
    assert np.all(r.to_numpy() == np.array([0, 1, 2, 3, 4]))


def test_tensor_indexing(t):
    assert t[0] == 1


# Reductions


def test_tensor_mean(t):
    r = t.mean()
    assert np.all(r.to_numpy() == t.to_numpy().mean())


def test_tensor_sum(t):
    r = t.sum()
    assert np.all(r.to_numpy() == t.to_numpy().sum())


def test_tensor_var(t):
    r = t.var()
    assert np.all(r.to_numpy() == np.array(np.var(t.to_numpy())))


def test_tensor_std(t):
    r = t.std()
    assert np.all(r.to_numpy() == np.array(np.std(t.to_numpy())))


def test_tensor_quantile(t):
    r = t.quantile(0.25)
    expected = np.quantile(t.to_numpy(), 0.25)
    assert np.allclose(r.to_numpy(), expected)


def test_tensor_percentile(t):
    r = t.percentile(25)
    expected = np.percentile(t.to_numpy(), 25)
    assert np.allclose(r.to_numpy(), expected)


def test_tensor_min(t):
    assert t.min() == 1


def test_tensor_max(t):
    assert t.max() == 5
