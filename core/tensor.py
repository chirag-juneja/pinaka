import numpy as np


class Tensor:
    # TODO: Migrate to C++ or Rust later
    def __init__(self, data: list | np.ndarray, dtype=np.float64):
        if isinstance(data, Tensor):
            self.data = data.data
        else:
            self.data = np.array(data, dtype=dtype)
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, dtype={self.dtype})"

    def to_numpy(self):
        return self.data
