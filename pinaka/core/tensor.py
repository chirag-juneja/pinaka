import numpy as np


class Tensor:
    # TODO: Migrate to C++ or Rust later
    # TODO: more tensor ops
    def __init__(self, data: list | np.ndarray, dtype=np.float64):
        if isinstance(data, Tensor):
            self.data = data.data
        else:
            self.data = np.array(data, dtype=dtype)
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def zeros(self, n):
        return Tensor(np.zeros(n))

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, dtype={self.dtype})"

    def to_numpy(self):
        return self.data

    # Indexing

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.shape[0]

    def _to_data(self, x):
        return x.data if isinstance(x, Tensor) else x

    # Tensor Ops

    def __add__(self, other):
        return Tensor(self.data + self._to_data(other))

    def __mul__(self, other):
        return Tensor(self.data * self._to_data(other))

    def __sub__(self, other):
        return Tensor(self.data - self._to_data(other))

    def __pow__(self, power):
        return Tensor(self.data**power)

    def __matmul__(self, other):
        return Tensor(self.data @ self._to_data(other))

    # Comparision
    def __eq__(self, other):
        return Tensor(self.data == self._to_data(other))

    def __ne__(self, other):
        return Tensor(self.data != self._to_data(other))

    def any(self):
        return bool(self.data.any())

    def all(self):
        return bool(self.data.all())

    # Stats

    def sum(self, axis=None):
        return Tensor(self.data.sum(axis=axis))

    def mean(self, axis=None):
        return Tensor(self.data.mean(axis=axis))

    def var(self, axis=None):
        m = self.mean(axis=axis)
        return ((self - m) ** 2).mean(axis=axis)

    def std(self, axis=None):
        v = self.var(axis=axis)
        return v**0.5

    def min(self, axis=None):
        return np.min(self.data, axis=axis)

    def max(self, axis=None):
        return np.max(self.data, axis=axis)

    def quantile(self, q, axis=None):
        return Tensor(np.quantile(self.data, q, axis=axis))

    def percentile(self, p, axis=None):
        q = np.array(p) / 100
        return Tensor(self.quantile(q, axis=axis))

    def summary(self):
        return {
            "min": self.min(),
            "q1": self.quantile(0.25),
            "median": self.quantile(0.5),
            "q3": self.quantile(0.75),
            "max": self.max(),
        }
