# Pinaka — Full ML Framework Build & Learning Plan

> **Vision:** A Python-first ML framework combining the statistical breadth of scikit-learn,
> the deep learning engine of PyTorch, the interoperability of ONNX, and the serving layer of
> FastAPI — with a planned migration path from NumPy → C++ → CUDA.
>
> **Repo:** https://github.com/chirag-juneja/pinaka  
> **Current state:** v0.1.2, NumPy dependency, pytest, bare `pinaka/` package.

---

## Books & How to Use Them

| Book | Role in this project |
|---|---|
| **Practical Statistics for Data Science** | **Primary guide for Phase 1.** Covers EDA, distributions, hypothesis testing, correlation, and regression — all implemented in `pinaka.stats` and `pinaka.feature`. Read a chapter, then implement it. |
| **Deep Learning from Scratch** | Primary guide for Phases 2 and 4. Read each chapter, then implement it in pinaka. |
| **ML from Scratch** (dafriedman97.github.io) | Primary guide for Phase 3. Their concept → math → code structure maps directly: your pinaka code replaces their implementation section. |
| **Hands-On ML with Scikit-Learn, Keras & TensorFlow** | Reference and breadth. Part 1 (Ch 1–8) for classical ML, Part 2 (Ch 10–16) for deep learning. Use to check understanding and find gaps. |

---

## Module Structure (target)

```
pinaka/
├── core/
│   ├── tensor.py          # Tensor class, backend dispatch
│   ├── backend_numpy.py   # NumPy backend (Phase 1)
│   ├── backend_cpp.py     # C++ backend via pybind11 (Phase 6)
│   └── ops.py             # Op registry — every op is a class with forward + backward
│
├── autograd/
│   ├── graph.py           # Node, Edge, computation graph
│   ├── engine.py          # backward(), topological sort
│   └── grad_check.py      # Numerical gradient checker
│
├── stats/                          # Phase 1 — statistics engine
│   ├── descriptive.py             # Mean, variance, std, skewness, kurtosis, moments
│   ├── distributions.py           # Normal, Bernoulli, Binomial, Poisson, t, chi-squared, F
│   ├── hypothesis.py              # t-test, z-test, chi-squared, ANOVA, Mann-Whitney U
│   ├── correlation.py             # Pearson, Spearman, Kendall tau, partial correlation, VIF
│   ├── information.py             # Entropy, KL divergence, mutual information, IV
│   ├── sampling.py                # Bootstrap, permutation tests, Monte Carlo
│   └── power.py                   # Statistical power, sample size estimation, effect sizes
│
├── feature/                        # Phase 1 — feature engineering
│   ├── selection.py               # Filter methods: correlation, chi-squared, mutual info, VIF
│   ├── extraction.py              # PCA, ICA, LDA, t-SNE, UMAP (linear + non-linear)
│   ├── construction.py            # Polynomial, interaction terms, binning, target encoding
│   ├── importance.py              # Permutation importance, SHAP values, Gini importance
│   └── validation.py              # Cross-validation, stratified k-fold, time-series split
│
├── models/                # Classical ML — the scikit-learn layer
│   ├── base.py            # Estimator base class: fit/predict/score
│   ├── linear.py          # LinearRegression, Ridge, Lasso, ElasticNet
│   ├── logistic.py        # LogisticRegression, SoftmaxRegression
│   ├── trees.py           # DecisionTree (classifier + regressor)
│   ├── ensemble.py        # RandomForest, GradientBoostedTrees
│   ├── naive_bayes.py     # GaussianNB, MultinomialNB
│   ├── svm.py             # SVM (linear kernel via dual formulation)
│   ├── decomposition.py   # PCA, SVD, TruncatedSVD
│   ├── clustering.py      # KMeans, DBSCAN, AgglomerativeClustering
│   └── neighbors.py       # KNN classifier + regressor
│
├── nn/                    # Deep learning — the PyTorch layer
│   ├── module.py          # Module base class
│   ├── layers.py          # Linear, Conv2d, ConvTranspose2d, Embedding
│   ├── normalization.py   # BatchNorm1d/2d, LayerNorm, GroupNorm
│   ├── recurrent.py       # RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU
│   ├── attention.py       # ScaledDotProductAttention, MultiHeadAttention
│   ├── transformer.py     # TransformerEncoderLayer, TransformerDecoderLayer
│   ├── activations.py     # ReLU, GELU, Sigmoid, Tanh, Softmax, Mish
│   ├── dropout.py         # Dropout, Dropout2d, AlphaDropout
│   ├── loss.py            # MSELoss, CrossEntropyLoss, BCELoss, HuberLoss
│   ├── init.py            # Kaiming, Xavier, orthogonal, normal init
│   └── containers.py      # Sequential, ModuleList, ModuleDict
│
├── optim/
│   ├── base.py            # Optimizer base class
│   ├── sgd.py             # SGD, SGD + Momentum, Nesterov
│   ├── adam.py            # Adam, AdamW, NAdam
│   ├── adagrad.py         # Adagrad, RMSProp, Adadelta
│   └── scheduler.py       # StepLR, CosineAnnealingLR, ReduceLROnPlateau
│
├── data/
│   ├── dataset.py         # Dataset base class
│   ├── dataloader.py      # DataLoader: batching, shuffling, collate
│   ├── transforms.py      # Normalize, RandomCrop, ToTensor, Compose
│   └── samplers.py        # RandomSampler, SequentialSampler, WeightedSampler
│
├── preprocessing/
│   ├── scalers.py         # StandardScaler, MinMaxScaler, RobustScaler
│   ├── encoders.py        # OneHotEncoder, LabelEncoder, OrdinalEncoder
│   ├── imputers.py        # SimpleImputer, IterativeImputer
│   └── pipeline.py        # Pipeline, ColumnTransformer, FeatureUnion
│
├── metrics/
│   ├── classification.py  # Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix
│   ├── regression.py      # MSE, MAE, R², MAPE, Huber
│   └── clustering.py      # Silhouette, Davies-Bouldin, Calinski-Harabasz
│
├── export/
│   ├── onnx_export.py     # Graph → ONNX serialization
│   ├── onnx_import.py     # ONNX → pinaka model
│   └── serialization.py   # save/load pinaka native format
│
└── serve/
    ├── server.py          # FastAPI ModelServer
    ├── schemas.py         # Pydantic schema auto-generation
    └── registry.py        # Model versioning and hot-swap
```

---

## Phase 1 — Tensor Engine + Statistics + Feature Engineering

**Goal:** Two things in parallel. First, build the computational substrate (the `Tensor` class and NumPy backend) that everything else runs on. Second — and equally important — build a full statistics and feature engineering library on top of it. This phase is where you develop the quantitative intuition that makes every later phase make sense.

**What you learn:** How NumPy's ndarray works. Broadcasting rules. The full vocabulary of descriptive statistics, probability distributions, hypothesis testing, correlation, and information theory. How to engineer, select, and validate features from raw data.

**Read first:** *Practical Statistics for Data Science* (cover to cover during this phase). Reference *Deep Learning from Scratch* Ch 1–2 for the tensor mechanics.

---

### 1.1 Backend Abstraction Protocol

Define a `TensorBackend` protocol before writing a single op. Every numerical operation goes through this interface — this is the design decision that makes Phase 6 (C++/CUDA) a backend swap rather than a rewrite.

```python
# pinaka/core/backend_numpy.py
from typing import Protocol

class TensorBackend(Protocol):
    def matmul(self, a, b): ...
    def add(self, a, b): ...
    def multiply(self, a, b): ...
    def sum(self, a, axis=None, keepdims=False): ...
    def reshape(self, a, shape): ...
    def transpose(self, a, axes=None): ...
    def maximum(self, a, b): ...
    def exp(self, a): ...
    def log(self, a): ...
    def sqrt(self, a): ...
    def zeros(self, shape, dtype=float): ...
    def ones(self, shape, dtype=float): ...
    def sort(self, a, axis=-1): ...
    def argsort(self, a, axis=-1): ...

class NumpyBackend:
    def matmul(self, a, b):              return np.matmul(a, b)
    def add(self, a, b):                 return np.add(a, b)
    def multiply(self, a, b):            return np.multiply(a, b)
    def sum(self, a, axis=None, keepdims=False): return np.sum(a, axis=axis, keepdims=keepdims)
    def reshape(self, a, shape):         return np.reshape(a, shape)
    def transpose(self, a, axes=None):   return np.transpose(a, axes)
    def maximum(self, a, b):             return np.maximum(a, b)
    def exp(self, a):                    return np.exp(a)
    def log(self, a):                    return np.log(a)
    def sqrt(self, a):                   return np.sqrt(a)
    def zeros(self, shape, dtype=float): return np.zeros(shape, dtype=dtype)
    def ones(self, shape, dtype=float):  return np.ones(shape, dtype=dtype)
    def sort(self, a, axis=-1):          return np.sort(a, axis=axis)
    def argsort(self, a, axis=-1):       return np.argsort(a, axis=axis)
```

**Never call `np.` directly in ops — always call `self._backend.method()`.** This is a discipline rule, not a suggestion. The C++ and CUDA backends replace exactly this file.

---

### 1.2 Tensor Class

```python
# pinaka/core/tensor.py
class Tensor:
    def __init__(self, data, requires_grad=False, dtype=None, device='cpu'):
        self._backend = _get_backend(device)   # NumpyBackend | CppBackend | CudaBackend
        self.data     = self._backend.asarray(data, dtype=dtype)
        self.shape    = self.data.shape
        self.dtype    = self.data.dtype
        self.device   = device

        # Autograd fields — set by ops, never by the user directly
        self.requires_grad = requires_grad
        self.grad          = None       # populated by backward()
        self._grad_fn      = None       # the Op that created this tensor
        self._inputs       = []         # input Tensors to that op

    def __repr__(self):
        grad_info = self._grad_fn.__class__.__name__ if self._grad_fn else None
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, grad_fn={grad_info})"

    # Operator overloading — delegate to ops
    def __add__(self, other):  return Add().forward(self, _ensure_tensor(other))
    def __mul__(self, other):  return Mul().forward(self, _ensure_tensor(other))
    def __matmul__(self, other): return MatMul().forward(self, _ensure_tensor(other))
    def __neg__(self):         return Neg().forward(self)
    def __sub__(self, other):  return self + (-_ensure_tensor(other))
    def __truediv__(self, other): return Div().forward(self, _ensure_tensor(other))
    def __pow__(self, n):      return Pow().forward(self, n)

    @property
    def T(self): return Transpose().forward(self)

    def sum(self, axis=None, keepdims=False): return Sum().forward(self, axis, keepdims)
    def mean(self, axis=None, keepdims=False): return Mean().forward(self, axis, keepdims)
    def reshape(self, *shape): return Reshape().forward(self, shape)

    def zero_grad(self): self.grad = None
    def item(self):      return self.data.item()
```

Key design decisions:
- `requires_grad=True` only on leaf tensors (model parameters). Intermediate tensors get `_grad_fn` set automatically by ops.
- `device` is a string (`'cpu'`) that drives backend dispatch. In Phase 6, `'cuda'` selects the CUDA backend.
- `grad` accumulates by default (like PyTorch). Always call `zero_grad()` before a new backward pass.

---

### 1.3 Ops to Implement

| Category | Ops |
|---|---|
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `pow`, `abs` |
| Matrix | `matmul`, `dot`, `outer`, `cross` |
| Reduction | `sum`, `mean`, `max`, `min`, `prod`, `var`, `std` |
| Shape | `reshape`, `transpose`, `squeeze`, `unsqueeze`, `flatten`, `expand`, `stack`, `concat` |
| Indexing | `__getitem__`, `slice`, `gather`, `scatter`, `where` |
| Math | `exp`, `log`, `log2`, `sqrt`, `abs`, `sign`, `floor`, `ceil`, `clip` |
| Sorting | `sort`, `argsort`, `topk` |
| Creation | `zeros`, `ones`, `eye`, `arange`, `linspace`, `rand`, `randn`, `full` |
| Type | `astype`, `to` (device), `item` |

---

### 1.4 Broadcasting

Broadcasting is where most tensor bugs live. Implement `_unbroadcast` carefully — it is used by every binary op's backward pass:

```python
def _unbroadcast(grad, target_shape):
    """Sum gradient over axes that were broadcast during the forward pass."""
    # If grad has more dimensions, sum over leading dims
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    # Sum over dims where target_shape has size 1 (the broadcast dims)
    for i, (gs, ts) in enumerate(zip(grad.shape, target_shape)):
        if ts == 1 and gs > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
```

Test all broadcasting patterns:
```python
def test_broadcast_scalar():    assert (Tensor([1,2,3]) + Tensor(1)).shape == (3,)
def test_broadcast_row():       assert (Tensor(np.ones((3,4))) + Tensor(np.ones((4,)))).shape == (3,4)
def test_broadcast_col():       assert (Tensor(np.ones((3,4))) + Tensor(np.ones((3,1)))).shape == (3,4)
def test_broadcast_batch():     assert (Tensor(np.ones((2,3,4))) + Tensor(np.ones((3,4)))).shape == (2,3,4)
```

---

### 1.5 Descriptive Statistics (`pinaka/stats/descriptive.py`)

Build the full vocabulary of descriptive statistics as a `Summary` class and standalone functions. Every function must use the Tensor backend — no raw NumPy calls.

**What you learn:** The precise mathematical definitions of statistics you may have used loosely before. The difference between population and sample variance. What skewness and kurtosis actually measure.

```python
# pinaka/stats/descriptive.py

class Summary:
    """Compute and store a complete statistical profile of a dataset."""

    def __init__(self, x: Tensor):
        self.n    = x.shape[0]
        self.mean = x.mean()
        self.std  = x.std()
        self.var  = x.var()
        self.min  = x.min()
        self.max  = x.max()
        self.range = self.max - self.min
        self.median   = median(x)
        self.q1, self.q3 = quantile(x, [0.25, 0.75])
        self.iqr      = self.q3 - self.q1
        self.skewness = skewness(x)
        self.kurtosis = kurtosis(x)

    def __repr__(self):
        return (f"Summary(n={self.n}, mean={self.mean:.4f}, std={self.std:.4f}, "
                f"skew={self.skewness:.4f}, kurt={self.kurtosis:.4f})")


def variance(x: Tensor, ddof: int = 1) -> Tensor:
    """Sample variance (ddof=1) or population variance (ddof=0)."""
    mu = x.mean()
    return ((x - mu) ** 2).sum() / (x.shape[0] - ddof)

def std(x: Tensor, ddof: int = 1) -> Tensor:
    return variance(x, ddof) ** 0.5

def skewness(x: Tensor) -> float:
    """
    Third standardized moment. Measures asymmetry of the distribution.
      skew > 0: right tail is heavier (most values left of mean)
      skew < 0: left tail is heavier
      skew = 0: symmetric
    """
    mu, s, n = x.mean(), std(x), x.shape[0]
    return float(((x - mu) ** 3).mean() / (s ** 3))

def kurtosis(x: Tensor, excess: bool = True) -> float:
    """
    Fourth standardized moment. Measures tail heaviness.
      excess=True: subtract 3 so Normal distribution gives kurtosis=0
      kurtosis > 0 (leptokurtic): heavier tails than Normal (e.g. t-distribution)
      kurtosis < 0 (platykurtic): lighter tails than Normal
    """
    mu, s = x.mean(), std(x)
    k = float(((x - mu) ** 4).mean() / (s ** 4))
    return k - 3 if excess else k

def median(x: Tensor) -> float:
    sorted_x = x.data.flatten()
    sorted_x.sort()
    n = len(sorted_x)
    return float(sorted_x[n//2] if n % 2 == 1 else (sorted_x[n//2-1] + sorted_x[n//2]) / 2)

def quantile(x: Tensor, q) -> list:
    """Compute quantiles using linear interpolation (same as NumPy default)."""
    return [float(np.quantile(x.data, qi)) for qi in (q if hasattr(q, '__iter__') else [q])]

def covariance_matrix(X: Tensor, ddof: int = 1) -> Tensor:
    """
    Cov(X) = (X - mu)^T (X - mu) / (n - ddof)
    Shape: (n_features, n_features). Diagonal = variances. Off-diagonal = covariances.
    """
    mu  = X.mean(axis=0)
    X_c = X - mu
    return (X_c.T @ X_c) / (X.shape[0] - ddof)

def standardize(X: Tensor, ddof: int = 1) -> Tensor:
    """Z-score: (x - mean) / std. Result has mean=0, std=1."""
    return (X - X.mean(axis=0)) / (X.std(axis=0, ddof=ddof) + 1e-8)

def winsorize(x: Tensor, lower: float = 0.05, upper: float = 0.95) -> Tensor:
    """Clip extreme values to given quantile bounds. Reduces outlier influence."""
    lo, hi = quantile(x, lower), quantile(x, upper)
    return x.clip(lo, hi)
```

---

### 1.6 Probability Distributions (`pinaka/stats/distributions.py`)

Implement each distribution with `pdf`, `cdf`, `log_pdf`, `sample`, and `fit_mle`. This is the bridge to Phase 3 — every ML loss function is a negative log-likelihood of one of these distributions.

**The connection to learn:** Minimizing MSE = MLE under a Gaussian. Minimizing binary cross-entropy = MLE under Bernoulli. Minimizing Poisson deviance = MLE under Poisson. Derive each from `log_pdf` before writing the loss function.

```python
# pinaka/stats/distributions.py

class Normal:
    """
    N(mu, sigma^2). The most important distribution in statistics.
    Appears everywhere via the Central Limit Theorem.
    """
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu, self.sigma = mu, sigma

    def pdf(self, x):
        z = (x - self.mu) / self.sigma
        return np.exp(-0.5 * z**2) / (self.sigma * np.sqrt(2 * np.pi))

    def log_pdf(self, x):
        """log N(x; mu, sigma) = -0.5*log(2*pi) - log(sigma) - (x-mu)^2 / (2*sigma^2)"""
        return -0.5 * np.log(2 * np.pi) - np.log(self.sigma) - 0.5 * ((x - self.mu)/self.sigma)**2

    def cdf(self, x):
        from scipy.special import erf
        return 0.5 * (1 + erf((x - self.mu) / (self.sigma * np.sqrt(2))))

    def log_likelihood(self, data) -> float:
        return float(self.log_pdf(data).sum())

    def fit_mle(self, data):
        """MLE for Normal: mu_hat = sample mean, sigma_hat = sample std."""
        self.mu    = float(data.mean())
        self.sigma = float(data.std())
        return self

    def sample(self, size=1):
        return np.random.normal(self.mu, self.sigma, size)

    def confidence_interval(self, n: int, alpha: float = 0.05):
        """95% CI for the mean given n observations: mu +/- z * sigma/sqrt(n)"""
        z = 1.96 if alpha == 0.05 else self._z_score(1 - alpha/2)
        margin = z * self.sigma / np.sqrt(n)
        return self.mu - margin, self.mu + margin


class StudentT:
    """
    t(nu) — heavier tails than Normal. Used when sigma is unknown and estimated from data.
    As nu -> inf, converges to Normal. Critical for hypothesis tests with small samples.
    """
    def __init__(self, df: float):
        self.df = df

    def pdf(self, t):
        from scipy.special import gamma
        nu = self.df
        return (gamma((nu+1)/2) / (np.sqrt(nu*np.pi) * gamma(nu/2))) * (1 + t**2/nu)**(-(nu+1)/2)

    def cdf(self, t):
        from scipy.stats import t as scipy_t
        return scipy_t.cdf(t, self.df)

    def ppf(self, p):
        """Percent-point function (inverse CDF) — used for critical values."""
        from scipy.stats import t as scipy_t
        return scipy_t.ppf(p, self.df)


class Bernoulli:
    """P(X=1) = p. Models binary outcomes. MLE -> logistic regression."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def log_pdf(self, x):
        """log P(x; p) = x*log(p) + (1-x)*log(1-p). This IS binary cross-entropy."""
        return x * np.log(self.p + 1e-10) + (1-x) * np.log(1 - self.p + 1e-10)

    def fit_mle(self, data):
        self.p = float(data.mean())
        return self


class ChiSquared:
    """chi^2(k). Sum of k squared standard normals. Used for goodness-of-fit tests."""
    def __init__(self, df: int):
        self.df = df

    def cdf(self, x):
        from scipy.stats import chi2
        return chi2.cdf(x, self.df)

    def ppf(self, p):
        from scipy.stats import chi2
        return chi2.ppf(p, self.df)


class Poisson:
    """P(X=k) = lambda^k * e^-lambda / k!. Models counts per interval."""
    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def log_pmf(self, k):
        return k * np.log(self.lam) - self.lam - np.array([np.math.lgamma(ki+1) for ki in k])

    def fit_mle(self, data):
        self.lam = float(data.mean())   # MLE for Poisson: lambda_hat = sample mean
        return self
```

---

### 1.7 Hypothesis Testing (`pinaka/stats/hypothesis.py`)

**What you learn:** How to make statistically rigorous comparisons. What a p-value actually means (and what it doesn't). When to use parametric vs non-parametric tests. The relationship between CIs and hypothesis tests.

Implement each test from scratch: compute the test statistic, compute the p-value using the appropriate distribution, return a structured result.

```python
# pinaka/stats/hypothesis.py
from dataclasses import dataclass

@dataclass
class TestResult:
    test_name:   str
    statistic:   float
    p_value:     float
    df:          float | None
    reject_h0:   bool
    alpha:       float
    effect_size: float | None = None
    ci:          tuple | None = None

    def __repr__(self):
        verdict = "REJECT H0" if self.reject_h0 else "FAIL TO REJECT H0"
        return (f"{self.test_name}: stat={self.statistic:.4f}, "
                f"p={self.p_value:.4f}, {verdict} (alpha={self.alpha})")


def one_sample_t_test(x: Tensor, mu0: float = 0.0, alpha: float = 0.05) -> TestResult:
    """
    H0: population mean == mu0
    H1: population mean != mu0

    t = (x_bar - mu0) / (s / sqrt(n))

    Use when: testing whether a sample mean differs from a known value.
    Example: does our model's error have zero mean?
    """
    n     = x.shape[0]
    x_bar = float(x.mean())
    s     = float(std(x))
    t_stat = (x_bar - mu0) / (s / np.sqrt(n))
    df    = n - 1
    dist  = StudentT(df)
    # Two-tailed: p = 2 * P(T > |t|)
    p_val = 2 * (1 - dist.cdf(abs(t_stat)))
    ci    = (x_bar - dist.ppf(1-alpha/2) * s/np.sqrt(n),
             x_bar + dist.ppf(1-alpha/2) * s/np.sqrt(n))
    return TestResult("One-sample t-test", t_stat, p_val, df, p_val < alpha, alpha, ci=ci)


def two_sample_t_test(x1: Tensor, x2: Tensor, equal_var: bool = True,
                      alpha: float = 0.05) -> TestResult:
    """
    H0: mu1 == mu2
    H1: mu1 != mu2

    Welch's t-test (equal_var=False) is generally safer — don't assume equal variances.

    Use when: comparing means of two independent groups.
    Example: does treatment group differ from control? Do two feature distributions differ?
    """
    n1, n2 = x1.shape[0], x2.shape[0]
    m1, m2 = float(x1.mean()), float(x2.mean())
    v1, v2 = float(variance(x1)), float(variance(x2))

    if equal_var:
        sp2 = ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)
        t_stat = (m1 - m2) / np.sqrt(sp2 * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:  # Welch
        se = np.sqrt(v1/n1 + v2/n2)
        t_stat = (m1 - m2) / se
        df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

    p_val = 2 * (1 - StudentT(df).cdf(abs(t_stat)))
    # Cohen's d effect size
    pooled_std = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    cohens_d   = (m1 - m2) / pooled_std
    return TestResult("Two-sample t-test (Welch)" if not equal_var else "Two-sample t-test",
                      t_stat, p_val, df, p_val < alpha, alpha, effect_size=cohens_d)


def paired_t_test(x1: Tensor, x2: Tensor, alpha: float = 0.05) -> TestResult:
    """
    H0: mean(x1 - x2) == 0. Equivalent to one-sample t-test on the differences.
    Use when: same subjects measured twice (before/after, two conditions).
    Example: does fine-tuning improve model performance on the same test set?
    """
    return one_sample_t_test(x1 - x2, mu0=0.0, alpha=alpha)


def chi_squared_test(observed: Tensor, expected: Tensor = None,
                     alpha: float = 0.05) -> TestResult:
    """
    Goodness-of-fit: H0: observed frequencies match expected.
    Independence: H0: two categorical variables are independent.

    chi^2 = sum((O - E)^2 / E)

    Use when: testing whether categorical distributions match, or whether
    two categorical features are associated.
    Example: are class labels uniformly distributed? Is feature X independent of label Y?
    """
    obs = observed.data.flatten().astype(float)
    if expected is None:
        exp = np.full_like(obs, obs.sum() / len(obs))
    else:
        exp = expected.data.flatten().astype(float)
    chi2 = float(((obs - exp)**2 / exp).sum())
    df   = len(obs) - 1
    p_val = 1 - ChiSquared(df).cdf(chi2)
    return TestResult("Chi-squared test", chi2, p_val, df, p_val < alpha, alpha)


def anova_one_way(*groups: Tensor, alpha: float = 0.05) -> TestResult:
    """
    H0: all group means are equal (mu1 = mu2 = ... = mk)

    F = (between-group variance) / (within-group variance)

    Use when: comparing means across 3+ groups.
    Example: does a feature's distribution differ across multiple classes?
    If significant, follow up with post-hoc pairwise t-tests (Bonferroni corrected).
    """
    k    = len(groups)
    ns   = [g.shape[0] for g in groups]
    N    = sum(ns)
    grand_mean = sum(g.mean() * n for g, n in zip(groups, ns)) / N

    ss_between = sum(n * (g.mean() - grand_mean)**2 for g, n in zip(groups, ns))
    ss_within  = sum(((g - g.mean())**2).sum() for g in groups)
    df_between, df_within = k - 1, N - k
    F = (float(ss_between) / df_between) / (float(ss_within) / df_within)

    from scipy.stats import f as f_dist
    p_val = 1 - f_dist.cdf(F, df_between, df_within)
    return TestResult("One-way ANOVA", F, p_val, (df_between, df_within), p_val < alpha, alpha)


def mann_whitney_u(x1: Tensor, x2: Tensor, alpha: float = 0.05) -> TestResult:
    """
    Non-parametric alternative to two-sample t-test. No normality assumption.
    H0: distributions of x1 and x2 are identical.
    Use when data is ordinal, or clearly non-normal, or sample sizes are small.
    """
    from scipy.stats import mannwhitneyu
    stat, p_val = mannwhitneyu(x1.data, x2.data, alternative='two-sided')
    return TestResult("Mann-Whitney U", float(stat), float(p_val), None, p_val < alpha, alpha)


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Correct for multiple comparisons. When running k tests, use alpha/k per test.
    Critical when doing pairwise comparisons after ANOVA, or feature selection tests.
    """
    k = len(p_values)
    return [p < alpha / k for p in p_values]


def false_discovery_rate(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg FDR correction. Less conservative than Bonferroni.
    Controls expected proportion of false discoveries rather than family-wise error.
    Use for high-dimensional feature selection (many simultaneous tests).
    """
    k = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]
    thresholds = alpha * (np.arange(1, k+1) / k)
    reject_sorted = sorted_p <= thresholds
    # Find largest i where H0 is rejected, reject all smaller
    if reject_sorted.any():
        last = np.where(reject_sorted)[0].max()
        reject_sorted[:last+1] = True
    result = np.zeros(k, dtype=bool)
    result[sorted_idx] = reject_sorted
    return list(result)
```

---

### 1.8 Correlation Analysis (`pinaka/stats/correlation.py`)

**What you learn:** The difference between linear correlation (Pearson) and rank correlation (Spearman). When correlation is misleading (Anscombe's quartet, Simpson's paradox). How multicollinearity is detected and why it matters for linear models.

```python
# pinaka/stats/correlation.py

def pearson_r(x: Tensor, y: Tensor) -> tuple[float, float]:
    """
    Linear correlation coefficient in [-1, 1].
    r = Cov(x,y) / (sigma_x * sigma_y)
    Assumptions: linear relationship, no outliers, both variables roughly normal.
    Returns: (r, p_value)
    """
    n    = x.shape[0]
    x_c  = x - x.mean()
    y_c  = y - y.mean()
    r    = float((x_c * y_c).sum() / (((x_c**2).sum() * (y_c**2).sum()) ** 0.5))
    # t-statistic for H0: rho=0
    t    = r * np.sqrt(n - 2) / np.sqrt(1 - r**2 + 1e-10)
    p    = 2 * (1 - StudentT(n-2).cdf(abs(t)))
    return r, p


def spearman_r(x: Tensor, y: Tensor) -> tuple[float, float]:
    """
    Rank correlation. Measures monotonic (not just linear) relationships.
    More robust to outliers than Pearson. Use for ordinal data or non-linear monotone relationships.
    Computed as Pearson on the ranks of x and y.
    """
    rx = Tensor(x.data.argsort().argsort().astype(float))
    ry = Tensor(y.data.argsort().argsort().astype(float))
    return pearson_r(rx, ry)


def kendall_tau(x: Tensor, y: Tensor) -> tuple[float, float]:
    """
    Rank correlation based on concordant/discordant pairs.
    tau = (concordant - discordant) / (n*(n-1)/2)
    More interpretable than Spearman: tau=0.3 means 30% more concordant than discordant pairs.
    """
    from scipy.stats import kendalltau
    tau, p = kendalltau(x.data, y.data)
    return float(tau), float(p)


def correlation_matrix(X: Tensor, method: str = 'pearson') -> Tensor:
    """
    Full p x p correlation matrix for a dataset with p features.
    Diagonal is always 1. Off-diagonal is pairwise correlation.
    Essential first step in EDA — shows feature relationships and collinearity.
    """
    p = X.shape[1]
    C = np.eye(p)
    fn = pearson_r if method == 'pearson' else spearman_r
    for i in range(p):
        for j in range(i+1, p):
            r, _ = fn(Tensor(X.data[:, i]), Tensor(X.data[:, j]))
            C[i, j] = C[j, i] = r
    return Tensor(C)


def partial_correlation(X: Tensor, i: int, j: int) -> float:
    """
    Correlation between features i and j after controlling for all other features.
    Removes the confounding effect of other variables.
    Use to distinguish direct correlations from spurious ones driven by a third variable.
    """
    # Regress feature i and j on all others, take correlation of residuals
    others = [k for k in range(X.shape[1]) if k not in (i, j)]
    X_ctrl = X.data[:, others]
    xi, xj = X.data[:, i], X.data[:, j]
    # Residuals from OLS regression
    def residuals(y, X_ctrl):
        beta = np.linalg.lstsq(X_ctrl, y, rcond=None)[0]
        return y - X_ctrl @ beta
    ri = residuals(xi, X_ctrl)
    rj = residuals(xj, X_ctrl)
    return float(pearson_r(Tensor(ri), Tensor(rj))[0])


def variance_inflation_factor(X: Tensor) -> Tensor:
    """
    VIF_j = 1 / (1 - R^2_j) where R^2_j is from regressing feature j on all other features.
    VIF > 5 signals moderate multicollinearity.
    VIF > 10 signals severe multicollinearity — consider removing the feature.
    Critical before fitting linear models: high VIF makes coefficients unstable.
    """
    p = X.shape[1]
    vifs = []
    for j in range(p):
        y = X.data[:, j]
        X_others = np.delete(X.data, j, axis=1)
        X_b = np.hstack([np.ones((len(X_others), 1)), X_others])
        y_hat = X_b @ np.linalg.lstsq(X_b, y, rcond=None)[0]
        ss_res = ((y - y_hat)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot
        vifs.append(1 / (1 - r2 + 1e-10))
    return Tensor(vifs)


def point_biserial(binary: Tensor, continuous: Tensor) -> tuple[float, float]:
    """
    Correlation between a binary and a continuous variable.
    Mathematically equivalent to Pearson r when one variable is binary.
    Use when: correlating a categorical feature (0/1 encoded) with a continuous target.
    """
    return pearson_r(binary.astype(float), continuous)
```

---

### 1.9 Information Theory (`pinaka/stats/information.py`)

**What you learn:** How entropy measures uncertainty. How mutual information generalizes correlation to non-linear relationships. How KL divergence measures the cost of using the wrong distribution. Why cross-entropy is the right loss for classification.

```python
# pinaka/stats/information.py

def entropy(p: Tensor, base: float = 2.0) -> float:
    """
    H(X) = -sum(p_i * log(p_i)).
    Measures the average uncertainty of a distribution.
    base=2: bits. base=e: nats.
    Uniform distribution has maximum entropy. Delta distribution has zero entropy.
    """
    p_safe = p.data + 1e-10
    return float(-(p_safe * np.log(p_safe) / np.log(base)).sum())


def cross_entropy(p: Tensor, q: Tensor) -> float:
    """
    H(p, q) = -sum(p_i * log(q_i)).
    Average code length when using distribution q to encode data from p.
    This IS the classification loss: p = true labels, q = predicted probabilities.
    H(p, q) = H(p) + KL(p || q), so minimizing CE minimizes KL divergence.
    """
    return float(-(p.data * np.log(q.data + 1e-10)).sum())


def kl_divergence(p: Tensor, q: Tensor) -> float:
    """
    KL(p || q) = sum(p_i * log(p_i / q_i)).
    Measures how much information is lost when q approximates p.
    Not symmetric: KL(p||q) != KL(q||p).
    Used in: VAEs, distillation, EM algorithm.
    """
    return float((p.data * np.log((p.data + 1e-10) / (q.data + 1e-10))).sum())


def mutual_information(X: Tensor, Y: Tensor, bins: int = 10) -> float:
    """
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    Measures shared information between two variables. Zero iff independent.
    Unlike correlation, captures non-linear relationships.
    Key use: feature selection — rank features by I(feature; target).
    """
    # Estimate joint distribution via histogram
    joint, _, _ = np.histogram2d(X.data.flatten(), Y.data.flatten(), bins=bins)
    joint = joint / joint.sum()   # normalize to probability
    px    = joint.sum(axis=1, keepdims=True)
    py    = joint.sum(axis=0, keepdims=True)
    mi    = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint[i,j] > 0:
                mi += joint[i,j] * np.log(joint[i,j] / (px[i,0] * py[0,j] + 1e-10))
    return float(mi)


def information_value(feature: Tensor, target: Tensor, bins: int = 10) -> float:
    """
    IV = sum((p_event - p_non_event) * WoE)  where WoE = log(dist_event / dist_non_event)
    Used in credit scoring and binary classification feature selection.
    IV < 0.02: not predictive. 0.02-0.1: weak. 0.1-0.3: medium. > 0.3: strong.
    """
    ...
```

---

### 1.10 Bootstrap & Sampling (`pinaka/stats/sampling.py`)

**What you learn:** How to estimate any statistic's sampling distribution without assuming normality. The difference between parametric and non-parametric inference. Why bootstrapping is one of the most practically useful techniques in statistics.

```python
# pinaka/stats/sampling.py

def bootstrap(x: Tensor, statistic_fn, n_boot: int = 1000,
              ci: float = 0.95) -> dict:
    """
    Estimate sampling distribution of any statistic via resampling with replacement.
    No distributional assumptions required.

    Use when:
    - Sample size is too small for CLT to apply reliably
    - Statistic has no closed-form SE (e.g., median, IQR, correlation)
    - You want CIs without assuming normality

    Returns the bootstrap distribution, SE, and percentile CI.
    """
    n     = x.shape[0]
    stats = []
    for _ in range(n_boot):
        idx    = np.random.randint(0, n, size=n)
        sample = Tensor(x.data[idx])
        stats.append(float(statistic_fn(sample)))

    stats  = np.array(stats)
    alpha  = (1 - ci) / 2
    return {
        'estimate':  float(statistic_fn(x)),
        'se':        float(stats.std()),
        'ci_lower':  float(np.quantile(stats, alpha)),
        'ci_upper':  float(np.quantile(stats, 1 - alpha)),
        'distribution': stats,
    }


def permutation_test(x1: Tensor, x2: Tensor, statistic_fn=None,
                     n_perm: int = 5000, alpha: float = 0.05) -> TestResult:
    """
    Non-parametric hypothesis test.
    H0: x1 and x2 come from the same distribution.
    Under H0, group labels are exchangeable — shuffle them and recompute statistic.
    p_value = fraction of permutations with statistic >= observed.

    More powerful than t-test when normality is violated.
    Ground truth for small n where asymptotic tests are unreliable.
    """
    if statistic_fn is None:
        statistic_fn = lambda a, b: float(a.mean()) - float(b.mean())

    observed = statistic_fn(x1, x2)
    combined = np.concatenate([x1.data, x2.data])
    n1 = x1.shape[0]
    null_dist = []

    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        null_dist.append(statistic_fn(Tensor(perm[:n1]), Tensor(perm[n1:])))

    p_val = float(np.mean(np.abs(null_dist) >= abs(observed)))
    return TestResult("Permutation test", observed, p_val, None, p_val < alpha, alpha)


def monte_carlo_pi(n: int = 1_000_000) -> float:
    """
    Estimate pi via Monte Carlo: throw random darts at a unit square,
    pi/4 = fraction landing in the unit circle.
    A simple first Monte Carlo exercise showing how sampling approximates integrals.
    """
    x, y = np.random.uniform(-1, 1, n), np.random.uniform(-1, 1, n)
    return 4 * float((x**2 + y**2 <= 1).mean())
```

---

### 1.11 Feature Engineering (`pinaka/feature/construction.py`)

**What you learn:** How raw features are rarely what a model needs. How domain knowledge and mathematical transformations create predictive signal. The techniques that close the gap between a baseline model and a competitive one.

```python
# pinaka/feature/construction.py

class PolynomialFeatures:
    """
    Expand features to include all polynomial terms up to degree d.
    x = [a, b] with degree=2 -> [1, a, b, a^2, ab, b^2]
    Use for: capturing non-linear relationships in linear models.
    Warning: with p features at degree d, produces C(p+d, d) features — explodes quickly.
    Always pair with regularization (Ridge/Lasso).
    """
    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def fit_transform(self, X: Tensor) -> Tensor:
        from itertools import combinations_with_replacement
        n, p = X.shape
        cols = [np.ones(n)] if self.include_bias else []
        for d in range(1, self.degree + 1):
            combos = combinations_with_replacement(range(p), d)
            if self.interaction_only and d > 1:
                combos = [c for c in combos if len(set(c)) == len(c)]
            for combo in combos:
                cols.append(np.prod(X.data[:, combo], axis=1))
        return Tensor(np.column_stack(cols))


class Binning:
    """
    Discretize a continuous variable into bins. Two strategies:
    - Equal-width: bins of equal range. Simple but sensitive to outliers.
    - Equal-frequency (quantile): each bin has same number of samples. More robust.
    Use for: capturing threshold effects, reducing outlier impact, making features
    compatible with tree-based models, creating ordinal features.
    """
    def __init__(self, n_bins: int = 5, strategy: str = 'quantile'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.edges_ = None

    def fit(self, x: Tensor):
        if self.strategy == 'quantile':
            self.edges_ = np.quantile(x.data, np.linspace(0, 1, self.n_bins + 1))
        elif self.strategy == 'uniform':
            self.edges_ = np.linspace(x.data.min(), x.data.max(), self.n_bins + 1)
        return self

    def transform(self, x: Tensor) -> Tensor:
        return Tensor(np.digitize(x.data, self.edges_[1:-1]).astype(float))

    def fit_transform(self, x): return self.fit(x).transform(x)


class TargetEncoder:
    """
    Replace each category with the mean of the target within that category.
    Much more powerful than one-hot for high-cardinality categoricals.
    Critical: use cross-fitting (compute encoding on held-out fold) to prevent
    target leakage. Without this, the encoder memorizes the target.
    """
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing   # shrinkage towards global mean for rare categories
        self.global_mean_ = None
        self.mapping_ = {}

    def fit(self, categories: np.ndarray, y: Tensor):
        self.global_mean_ = float(y.mean())
        for cat in np.unique(categories):
            mask = categories == cat
            n_cat   = mask.sum()
            cat_mean = float(y.data[mask].mean())
            # Smoothed estimate: blend category mean with global mean
            # as n_cat -> 0, weight shifts to global mean
            weight = n_cat / (n_cat + self.smoothing)
            self.mapping_[cat] = weight * cat_mean + (1 - weight) * self.global_mean_
        return self

    def transform(self, categories: np.ndarray) -> Tensor:
        return Tensor(np.array([self.mapping_.get(c, self.global_mean_) for c in categories]))


class CyclicalEncoder:
    """
    Encode cyclical features (hour, day of week, month, angle) as sin/cos pairs.
    hour=0 and hour=23 are adjacent but numerically far apart — sin/cos fixes this.
    Produces two features that correctly capture the cyclic distance.
    """
    def __init__(self, period: float):
        self.period = period

    def transform(self, x: Tensor) -> Tensor:
        angle = 2 * np.pi * x.data / self.period
        return Tensor(np.column_stack([np.sin(angle), np.cos(angle)]))


class LogTransform:
    """
    Apply log(1 + x) to right-skewed features (income, price, count data).
    Compresses large values, makes distribution more symmetric.
    Only valid for non-negative values. Use log1p to handle zeros.
    """
    def transform(self, x: Tensor) -> Tensor:
        return Tensor(np.log1p(x.data))

    def inverse_transform(self, x: Tensor) -> Tensor:
        return Tensor(np.expm1(x.data))
```

---

### 1.12 Feature Selection (`pinaka/feature/selection.py`)

**What you learn:** How to distinguish signal from noise in high-dimensional data. The three families of feature selection: filter (rank independently of model), wrapper (use model performance), embedded (regularization selects during training). Why more features isn't always better.

```python
# pinaka/feature/selection.py

class FilterSelector:
    """
    Rank features by a statistical score computed independently of any model.
    Fast: O(n*p), no model training required.
    Limitation: ignores feature interactions and redundancy between features.
    """
    def __init__(self, score_fn: str = 'mutual_info', k: int = 10):
        self.score_fn = score_fn
        self.k = k
        self.scores_ = None
        self.selected_ = None

    def fit(self, X: Tensor, y: Tensor):
        p = X.shape[1]
        scores = np.zeros(p)
        for j in range(p):
            xj = Tensor(X.data[:, j])
            if self.score_fn == 'mutual_info':
                scores[j] = mutual_information(xj, y)
            elif self.score_fn == 'pearson':
                scores[j] = abs(pearson_r(xj, y)[0])
            elif self.score_fn == 'spearman':
                scores[j] = abs(spearman_r(xj, y)[0])
            elif self.score_fn == 'chi2':
                scores[j] = chi_squared_test(xj, y).statistic
        self.scores_   = scores
        self.selected_ = np.argsort(scores)[::-1][:self.k]
        return self

    def transform(self, X: Tensor) -> Tensor:
        return Tensor(X.data[:, self.selected_])

    def fit_transform(self, X, y): return self.fit(X, y).transform(X)


class CorrelationSelector:
    """
    Remove features that are:
    1. Too correlated with each other (redundant — pick one)
    2. Too weakly correlated with the target (uninformative)

    More principled than univariate filtering — accounts for redundancy.
    """
    def __init__(self, target_threshold: float = 0.1,
                 feature_threshold: float = 0.9):
        self.target_threshold  = target_threshold
        self.feature_threshold = feature_threshold
        self.selected_ = None

    def fit(self, X: Tensor, y: Tensor):
        p = X.shape[1]
        # Step 1: drop features with low correlation to target
        target_corrs = np.array([abs(pearson_r(Tensor(X.data[:,j]), y)[0]) for j in range(p)])
        informative  = np.where(target_corrs >= self.target_threshold)[0]

        # Step 2: among informative features, remove redundant ones
        X_info = X.data[:, informative]
        C = np.corrcoef(X_info.T)
        keep = []
        for i in range(len(informative)):
            if all(abs(C[i, j]) < self.feature_threshold for j in keep):
                keep.append(i)

        self.selected_ = informative[keep]
        return self

    def transform(self, X: Tensor) -> Tensor:
        return Tensor(X.data[:, self.selected_])


class RecursiveFeatureEliminator:
    """
    Wrapper method: train model, remove least important feature, repeat.
    Expensive (fits model k times) but captures feature interactions.
    Use with models that provide feature importance (trees, linear coefficients).
    """
    def __init__(self, estimator, n_features: int = 10, step: int = 1):
        self.estimator  = estimator
        self.n_features = n_features
        self.step       = step
        self.ranking_   = None
        self.selected_  = None

    def fit(self, X: Tensor, y: Tensor):
        n_features = X.shape[1]
        support    = np.ones(n_features, dtype=bool)
        ranking    = np.ones(n_features, dtype=int)
        current_X  = X.data.copy()
        current_idx = np.arange(n_features)

        while current_idx.sum() > self.n_features:
            self.estimator.fit(Tensor(current_X), y)
            importances = self._get_importances()
            # Remove least important `step` features
            worst = np.argsort(importances)[:self.step]
            ranking[current_idx[worst]] = current_idx.sum()
            current_X   = np.delete(current_X, worst, axis=1)
            current_idx = np.delete(current_idx, worst)

        self.selected_ = current_idx
        self.ranking_  = ranking
        return self

    def _get_importances(self):
        if hasattr(self.estimator, 'coef_'):
            return np.abs(self.estimator.coef_)
        elif hasattr(self.estimator, 'feature_importances_'):
            return self.estimator.feature_importances_
        raise ValueError("Estimator has no coef_ or feature_importances_")

    def transform(self, X: Tensor) -> Tensor:
        return Tensor(X.data[:, self.selected_])
```

---

### 1.13 Cross-Validation (`pinaka/feature/validation.py`)

**What you learn:** Why a train/test split is not enough. How cross-validation gives you an unbiased estimate of generalization error. Why data leakage from preprocessing is a common and serious mistake.

```python
# pinaka/feature/validation.py

def k_fold_cv(estimator, X: Tensor, y: Tensor, k: int = 5,
              shuffle: bool = True, random_state: int = 42) -> dict:
    """
    K-fold cross-validation. Splits data into k folds; train on k-1, evaluate on 1.
    Repeat k times. Reports mean and std of score across folds.

    Important: the estimator is fit fresh on each fold — never reuse a fitted estimator.
    Critical mistake to avoid: don't fit preprocessors (scalers, encoders) on the
    whole dataset before CV. Fit them inside each fold on the training portion only.
    """
    np.random.seed(random_state)
    n     = X.shape[0]
    idx   = np.random.permutation(n) if shuffle else np.arange(n)
    folds = np.array_split(idx, k)
    scores = []

    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_tr, y_tr = Tensor(X.data[train_idx]), Tensor(y.data[train_idx])
        X_val, y_val = Tensor(X.data[val_idx]), Tensor(y.data[val_idx])

        from copy import deepcopy
        est = deepcopy(estimator)
        est.fit(X_tr, y_tr)
        scores.append(est.score(X_val, y_val))

    return {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}


def stratified_k_fold(estimator, X: Tensor, y: Tensor, k: int = 5) -> dict:
    """
    Stratified CV: each fold preserves the class distribution of the full dataset.
    Essential for imbalanced classification — prevents a fold from having no minority samples.
    """
    classes, counts = np.unique(y.data, return_counts=True)
    fold_idx = [[] for _ in range(k)]
    for cls in classes:
        cls_idx = np.where(y.data == cls)[0]
        np.random.shuffle(cls_idx)
        for i, chunk in enumerate(np.array_split(cls_idx, k)):
            fold_idx[i].extend(chunk.tolist())
    # Run CV with these stratified folds
    scores = []
    for i in range(k):
        val_idx = np.array(fold_idx[i])
        train_idx = np.concatenate([fold_idx[j] for j in range(k) if j != i])
        X_tr, y_tr = Tensor(X.data[train_idx]), Tensor(y.data[train_idx])
        X_val, y_val = Tensor(X.data[val_idx]), Tensor(y.data[val_idx])
        from copy import deepcopy
        est = deepcopy(estimator)
        est.fit(X_tr, y_tr)
        scores.append(est.score(X_val, y_val))
    return {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}


def time_series_cv(estimator, X: Tensor, y: Tensor, n_splits: int = 5,
                   gap: int = 0) -> dict:
    """
    Walk-forward validation for time-series data.
    Training always precedes validation — never look into the future.
    gap: number of samples to exclude between train and val (for multi-step-ahead forecasting).
    """
    n      = X.shape[0]
    size   = n // (n_splits + 1)
    scores = []
    for i in range(1, n_splits + 1):
        train_end = i * size
        val_start = train_end + gap
        val_end   = val_start + size
        if val_end > n: break
        X_tr  = Tensor(X.data[:train_end])
        y_tr  = Tensor(y.data[:train_end])
        X_val = Tensor(X.data[val_start:val_end])
        y_val = Tensor(y.data[val_start:val_end])
        from copy import deepcopy
        est = deepcopy(estimator)
        est.fit(X_tr, y_tr)
        scores.append(est.score(X_val, y_val))
    return {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}
```

---

### 1.14 Phase 1 Tests

```python
# tests/unit/test_tensor_ops.py
def test_matmul_matches_numpy():
    a = Tensor([[1., 2.], [3., 4.]])
    b = Tensor([[5., 6.], [7., 8.]])
    assert np.allclose((a @ b).data, np.array([[1,2],[3,4]]) @ np.array([[5,6],[7,8]]))

def test_broadcasting_shapes():
    assert (Tensor(np.ones((3,4))) + Tensor(np.ones((4,)))).shape == (3, 4)
    assert (Tensor(np.ones((2,3,4))) + Tensor(np.ones((3,4)))).shape == (2, 3, 4)

# tests/unit/test_stats.py
def test_variance_ddof():
    x = Tensor([2., 4., 4., 4., 5., 5., 7., 9.])
    assert abs(float(variance(x, ddof=1)) - 4.571) < 0.01
    assert abs(float(variance(x, ddof=0)) - 4.0)   < 0.01

def test_skewness_normal():
    np.random.seed(0)
    x = Tensor(np.random.normal(0, 1, 10000))
    assert abs(skewness(x)) < 0.1    # ~0 for symmetric distribution

def test_pearson_known():
    x = Tensor([1.,2.,3.,4.,5.])
    y = Tensor([2.,4.,6.,8.,10.])   # perfect linear relationship
    r, p = pearson_r(x, y)
    assert abs(r - 1.0) < 1e-10
    assert p < 0.001

def test_t_test_null():
    np.random.seed(42)
    x = Tensor(np.random.normal(0, 1, 100))
    result = one_sample_t_test(x, mu0=0.0)
    assert not result.reject_h0    # should not reject H0 (data IS from mu=0)

def test_t_test_alternative():
    np.random.seed(42)
    x = Tensor(np.random.normal(5, 1, 50))
    result = one_sample_t_test(x, mu0=0.0)
    assert result.reject_h0        # should reject H0 (data is clearly from mu=5)

def test_chi_squared_uniform():
    obs = Tensor([25., 25., 25., 25.])   # perfectly uniform
    result = chi_squared_test(obs)
    assert result.p_value > 0.05   # cannot reject uniform H0

def test_vif_independent():
    np.random.seed(0)
    X = Tensor(np.random.randn(200, 3))    # independent features -> VIF ~1
    vifs = variance_inflation_factor(X)
    assert all(v < 2.0 for v in vifs.data)

def test_mutual_info_independent():
    np.random.seed(0)
    x = Tensor(np.random.randn(1000))
    y = Tensor(np.random.randn(1000))
    assert mutual_information(x, y) < 0.1    # near zero for independent variables

def test_bootstrap_mean():
    np.random.seed(0)
    x = Tensor(np.random.normal(5, 2, 200))
    result = bootstrap(x, lambda t: float(t.mean()), n_boot=2000)
    assert result['ci_lower'] < 5.0 < result['ci_upper']

def test_polynomial_features_shape():
    X = Tensor(np.ones((10, 2)))
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X)
    assert Xp.shape == (10, 6)    # [1, x1, x2, x1^2, x1*x2, x2^2]

def test_target_encoder_no_leakage():
    cats = np.array(['a','b','a','b','a'])
    y    = Tensor([1.,2.,1.,2.,1.])
    enc  = TargetEncoder(smoothing=0)
    enc.fit(cats, y)
    out = enc.transform(cats)
    assert abs(float(out.data[0]) - 1.0) < 0.01   # category 'a' mean = 1.0

def test_kfold_cv_score_range():
    from pinaka.models.linear import LinearRegression
    np.random.seed(0)
    X = Tensor(np.random.randn(100, 5))
    y = Tensor(X.data @ np.array([1,2,3,4,5]) + np.random.randn(100)*0.1)
    result = k_fold_cv(LinearRegression(), X, y, k=5)
    assert result['mean'] > 0.95    # should fit very well on this linear problem
```

---

**Phase 1 Milestones:**

1. **Tensor ops** — every op matches NumPy to `np.allclose` tolerance. All broadcasting shapes are correct.
2. **Descriptive stats** — `Summary` on any 1D or 2D array matches scipy/pandas output exactly.
3. **Distributions** — `fit_mle` on 10,000 samples recovers true parameters within 1%.
4. **Hypothesis tests** — all tests produce correct verdicts on both null and alternative cases. p-values match scipy.stats within 0.001.
5. **Correlation** — Pearson/Spearman/Kendall match scipy. VIF correctly flags correlated features.
6. **Bootstrap** — 95% CI contains true parameter in ~95% of simulations over 100 runs.
7. **Feature engineering** — `PolynomialFeatures` shapes match sklearn. `TargetEncoder` is leakage-free.
8. **Cross-validation** — `k_fold_cv` on a linear problem achieves R² > 0.95.

---

## Phase 2 — Automatic Differentiation

**Goal:** Implement reverse-mode autograd. This is the single most important thing to understand in deep learning — if you understand this phase, everything else follows.

**What you learn:** The chain rule expressed as a graph traversal algorithm. Why every op must store its inputs. How gradient flow actually works.

**Read first:** *Deep Learning from Scratch*, Chapters 4–5.

### 2.1 The Core Insight

During a forward pass through a computation graph, every op records:
1. Its input tensors (to know which nodes to send gradients back to)
2. Any intermediate values needed to compute the gradient (e.g., `ReLU` needs to remember which inputs were positive)

The backward pass is a topological sort of the graph in reverse, calling each op's `backward()` with the incoming gradient.

### 2.2 Topological Sort

```python
# pinaka/autograd/engine.py
def backward(root: Tensor):
    """Reverse-mode automatic differentiation from root tensor."""
    # Build topological order
    topo = []
    visited = set()

    def build_topo(node):
        if id(node) not in visited:
            visited.add(id(node))
            for inp in node._inputs:
                build_topo(inp)
            topo.append(node)

    build_topo(root)

    # Seed gradient at root (scalar output assumed)
    root.grad = Tensor(np.ones_like(root.data))

    # Propagate gradients in reverse topological order
    for node in reversed(topo):
        if node._grad_fn is None:
            continue  # leaf node — gradient already accumulated
        grads = node._grad_fn.backward(node.grad)
        if not isinstance(grads, tuple):
            grads = (grads,)
        for inp, g in zip(node._inputs, grads):
            if inp.requires_grad:
                inp.grad = g if inp.grad is None else inp.grad + g
```

### 2.3 Gradient Rules for Every Op

These are the mathematical derivatives you need to implement in each op's `backward()`:

| Op | Forward | Backward (gradient of inputs) |
|---|---|---|
| `add(a, b)` | `a + b` | `grad_a = grad`, `grad_b = grad` |
| `mul(a, b)` | `a * b` | `grad_a = grad * b`, `grad_b = grad * a` |
| `matmul(a, b)` | `a @ b` | `grad_a = grad @ b.T`, `grad_b = a.T @ grad` |
| `sum(a)` | `Σa` | `grad_a = broadcast(grad, a.shape)` |
| `mean(a)` | `Σa/n` | `grad_a = broadcast(grad/n, a.shape)` |
| `exp(a)` | `eᵃ` | `grad_a = grad * exp(a)` (save `exp(a)` in forward) |
| `log(a)` | `log(a)` | `grad_a = grad / a` (save `a` in forward) |
| `relu(a)` | `max(0, a)` | `grad_a = grad * (a > 0)` |
| `sigmoid(a)` | `1/(1+e⁻ᵃ)` | `grad_a = grad * s * (1 - s)` (save `s=sigmoid(a)`) |
| `pow(a, n)` | `aⁿ` | `grad_a = grad * n * a^(n-1)` |
| `reshape(a)` | shape change | `grad_a = reshape(grad, a.shape)` |
| `transpose(a)` | axes permute | `grad_a = transpose(grad, inverse_axes)` |

### 2.4 Saving Intermediates

Some ops need to save values from the forward pass to compute gradients:

```python
class ReLU:
    def forward(self, a: Tensor) -> Tensor:
        self._mask = a.data > 0    # save which inputs were positive
        out = Tensor(np.maximum(a.data, 0))
        out._grad_fn = self
        out._inputs  = [a]
        return out

    def backward(self, grad_output):
        return Tensor(grad_output.data * self._mask)   # zero out negative positions

class Sigmoid:
    def forward(self, a: Tensor) -> Tensor:
        s = 1 / (1 + np.exp(-a.data))
        self._s = s                # save sigmoid output — needed in backward
        out = Tensor(s)
        out._grad_fn = self
        out._inputs  = [a]
        return out

    def backward(self, grad_output):
        return Tensor(grad_output.data * self._s * (1 - self._s))
```

### 2.5 Numerical Gradient Checker

This is the most important test you can write. It verifies your analytical gradient against the definition of a derivative, with no assumptions about your backward implementation:

```python
# pinaka/autograd/grad_check.py
def numerical_gradient(f, inputs, eps=1e-5):
    """Compute numerical Jacobian for each input via finite differences."""
    grads = []
    for inp in inputs:
        grad = np.zeros_like(inp.data)
        it = np.nditer(inp.data, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig = inp.data[idx]

            inp.data[idx] = orig + eps
            f_plus = f(*inputs).data.sum()

            inp.data[idx] = orig - eps
            f_minus = f(*inputs).data.sum()

            grad[idx] = (f_plus - f_minus) / (2 * eps)
            inp.data[idx] = orig
            it.iternext()
        grads.append(grad)
    return grads

def check_gradients(f, inputs, eps=1e-5, tol=1e-4):
    """Run forward+backward and compare to numerical gradients."""
    # Forward + backward
    for inp in inputs:
        inp.requires_grad = True
    out = f(*inputs)
    out.backward()
    analytical = [inp.grad.data for inp in inputs]

    # Numerical
    numerical = numerical_gradient(f, inputs, eps)

    for i, (a, n) in enumerate(zip(analytical, numerical)):
        rel_error = np.abs(a - n) / (np.abs(a) + np.abs(n) + 1e-8)
        assert rel_error.max() < tol, \
            f"Input {i}: max relative error {rel_error.max():.2e} > {tol}"
    return True
```

```python
# tests/test_autograd.py
def test_matmul_backward():
    a = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
    b = Tensor([[5., 6.], [7., 8.]], requires_grad=True)
    check_gradients(lambda a, b: (a @ b).sum(), [a, b])

def test_sigmoid_backward():
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    check_gradients(lambda x: sigmoid(x).sum(), [x])
```

### 2.6 Additional autograd features

```python
class Tensor:
    def zero_grad(self):
        self.grad = None

    def detach(self):
        """Return a new Tensor with same data but no gradient tracking."""
        return Tensor(self.data.copy())

    def __enter__(self):   # for torch.no_grad()-style context managers
        ...
```

**Implement `no_grad()` context manager** — essential for inference and evaluation:

```python
# pinaka/autograd/engine.py
from contextlib import contextmanager

_grad_enabled = True

@contextmanager
def no_grad():
    global _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = True
```

**Milestone:** Numerical gradient check passes (relative error < 1e-4) for: add, mul, matmul, exp, log, relu, sigmoid, sum, mean, reshape, transpose. Manually trace a 3-node graph on paper and match pinaka's gradient output exactly.

---

## Phase 3 — Statistics & Classical ML

**Goal:** Build the scikit-learn layer. Implement every algorithm from statistical first principles.

**What you learn:** The probabilistic foundation of ML. Why MSE is the right loss for regression. Why cross-entropy is the right loss for classification. What regularization actually does statistically.

**Read first (per algorithm):**
- *Practical Statistics for Data Science* — chapter for that algorithm family
- *ML from Scratch* (dafriedman97) — concept + math sections

### 3.1 The Bridge: MLE Unifies Statistics and ML

Before implementing any model, understand this connection:

| Statistical view | ML view |
|---|---|
| MLE under Gaussian noise → OLS | Minimize MSE loss |
| MLE under Bernoulli → Logistic regression | Minimize cross-entropy loss |
| MAP with Gaussian prior → Ridge regression | L2 regularization |
| MAP with Laplace prior → Lasso regression | L1 regularization |

Every loss function in ML is a negative log-likelihood. Implementing them from this view gives you the intuition that is otherwise mysterious.

### 3.2 Estimator Base Class (sklearn-compatible)

```python
# pinaka/models/base.py
from abc import ABC, abstractmethod

class Estimator(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    def score(self, X, y):
        """Default: R² for regression, accuracy for classification."""
        raise NotImplementedError

    def get_params(self): return self.__dict__
    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

class Classifier(Estimator):
    def predict_proba(self, X): raise NotImplementedError

class Regressor(Estimator):
    def score(self, X, y):
        from pinaka.metrics.regression import r2_score
        return r2_score(y, self.predict(X))
```

### 3.3 Statistics Module

```python
# pinaka/stats/descriptive.py
class DescriptiveStats:
    @staticmethod
    def mean(x, axis=None):           return x.mean(axis=axis)
    @staticmethod
    def variance(x, ddof=1):          return ((x - x.mean())**2).sum() / (len(x) - ddof)
    @staticmethod
    def std(x, ddof=1):               return variance(x, ddof)**0.5
    @staticmethod
    def covariance_matrix(X, ddof=1):
        mu = X.mean(axis=0)
        X_c = X - mu
        return (X_c.T @ X_c) / (len(X) - ddof)
    @staticmethod
    def pearson_correlation(x, y):
        return ((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std())

# pinaka/stats/distributions.py
class Normal:
    def __init__(self, mu=0, sigma=1):
        self.mu, self.sigma = mu, sigma

    def pdf(self, x):
        return (1/(self.sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-self.mu)/self.sigma)**2)

    def log_pdf(self, x):
        return -np.log(self.sigma) - 0.5*np.log(2*np.pi) - 0.5*((x-self.mu)/self.sigma)**2

    def log_likelihood(self, data):
        return self.log_pdf(data).sum()

    def fit_mle(self, data):
        """MLE: mu=sample mean, sigma=sample std."""
        self.mu = data.mean()
        self.sigma = data.std()
        return self
```

### 3.4 Linear Models

**Linear Regression — two ways, understand both:**

```python
# pinaka/models/linear.py
class LinearRegression(Regressor):
    """OLS via normal equation (exact) or gradient descent (approximate)."""

    def fit(self, X, y, method='normal_equation'):
        X_b = np.hstack([np.ones((len(X), 1)), X])   # add bias column
        if method == 'normal_equation':
            # β = (XᵀX)⁻¹Xᵀy — exact closed form
            self.weights = np.linalg.lstsq(X_b, y, rcond=None)[0]
        elif method == 'gradient_descent':
            self._fit_gd(X_b, y)
        return self

    def _fit_gd(self, X, y, lr=0.01, epochs=1000):
        n, d = X.shape
        self.weights = np.zeros(d)
        for _ in range(epochs):
            y_pred = X @ self.weights
            grad = (2/n) * X.T @ (y_pred - y)   # ∂MSE/∂w = 2/n * Xᵀ(Xw - y)
            self.weights -= lr * grad

    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b @ self.weights

class Ridge(LinearRegression):
    """Ridge = OLS + L2 regularization. MAP with Gaussian prior."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        n, d = X_b.shape
        # β = (XᵀX + αI)⁻¹Xᵀy  — the regularization term makes XᵀX invertible
        A = X_b.T @ X_b + self.alpha * np.eye(d)
        A[0, 0] -= self.alpha  # don't regularize the bias
        self.weights = np.linalg.solve(A, X_b.T @ y)
        return self
```

### 3.5 Logistic Regression

```python
class LogisticRegression(Classifier):
    """Binary cross-entropy loss. Derives from MLE on Bernoulli distribution."""
    def __init__(self, lr=0.01, epochs=1000, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        self.weights = np.zeros(X_b.shape[1])
        for _ in range(self.epochs):
            p = self._sigmoid(X_b @ self.weights)
            # Gradient of -log P(y|X) = Xᵀ(p - y) / n  +  λw (L2 term)
            grad = X_b.T @ (p - y) / len(y)
            grad[1:] += self.l2 * self.weights[1:]    # don't regularize bias
            self.weights -= self.lr * grad
        return self

    def predict_proba(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return self._sigmoid(X_b @ self.weights)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
```

### 3.6 Decision Trees

```python
class DecisionTree(Estimator):
    """CART — Classification and Regression Trees."""
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion   # 'gini', 'entropy', 'mse'

    def _gini(self, y):
        n = len(y)
        return 1 - sum((np.sum(y==c)/n)**2 for c in np.unique(y))

    def _information_gain(self, y, left_mask):
        n = len(y)
        n_l, n_r = left_mask.sum(), (~left_mask).sum()
        return self._gini(y) - (n_l/n)*self._gini(y[left_mask]) - (n_r/n)*self._gini(y[~left_mask])

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                mask = X[:, feat] <= thresh
                if mask.sum() < self.min_samples_split:
                    continue
                gain = self._information_gain(y, mask)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return {'leaf': True, 'value': np.bincount(y).argmax()}
        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {'leaf': True, 'value': np.bincount(y).argmax()}
        mask = X[:, feat] <= thresh
        return {'leaf': False, 'feat': feat, 'thresh': thresh,
                'left': self._build(X[mask], y[mask], depth+1),
                'right': self._build(X[~mask], y[~mask], depth+1)}

    def fit(self, X, y):
        self.tree_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])
```

### 3.7 Preprocessing Pipeline

```python
# pinaka/preprocessing/pipeline.py
class Pipeline:
    def __init__(self, steps: list[tuple[str, Estimator]]):
        self.steps = steps

    def fit(self, X, y=None):
        X_t = X
        for name, step in self.steps[:-1]:    # all but the last
            X_t = step.fit_transform(X_t, y)
        self.steps[-1][1].fit(X_t, y)         # last step: just fit
        return self

    def predict(self, X):
        X_t = X
        for name, step in self.steps[:-1]:
            X_t = step.transform(X_t)
        return self.steps[-1][1].predict(X_t)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
```

### 3.8 Models to implement

| Model | Key concept to learn | Statistical derivation |
|---|---|---|
| `LinearRegression` | OLS, normal equation | MLE under Gaussian noise |
| `Ridge` | L2 regularization | MAP with Gaussian prior |
| `Lasso` | L1 regularization, sparsity | MAP with Laplace prior |
| `ElasticNet` | L1 + L2 combined | — |
| `LogisticRegression` | Sigmoid, binary CE | MLE on Bernoulli |
| `SoftmaxRegression` | Softmax, multiclass CE | MLE on Categorical |
| `GaussianNB` | Bayes' theorem | Generative model, Gaussian class-conditionals |
| `DecisionTree` | Information gain, Gini | — |
| `RandomForest` | Bagging, decorrelated trees | Variance reduction |
| `GradientBoostedTrees` | Boosting, additive models | Functional gradient descent |
| `PCA` | Eigenvectors, explained variance | SVD of covariance matrix |
| `KMeans` | EM algorithm | — |
| `KNN` | Distance metrics | Non-parametric density estimation |

**Milestone:** Train pinaka `LogisticRegression` on Iris → >97% accuracy. Train pinaka `RandomForest` on a Kaggle tabular dataset → within 1% of sklearn. PCA on MNIST reconstructs digits visually correctly.

---

## Phase 4 — Deep Learning (Neural Network Layer)

**Goal:** Build the PyTorch layer on top of the autograd engine from Phase 2.

**What you learn:** How a neural network is just a graph of reusable parameterized ops. Why the Module abstraction is so powerful. How to train anything.

**Read:** *Deep Learning from Scratch* (all chapters). *Hands-On ML* Part 2.

### 4.1 Module Base Class

```python
# pinaka/nn/module.py
class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        """Recursively yield all leaf Tensors with requires_grad=True."""
        for name, val in self.__dict__.items():
            if isinstance(val, Tensor) and val.requires_grad:
                yield val
            elif isinstance(val, Module):
                yield from val.parameters()
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        yield from item.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):   self._training = True;  return self
    def eval(self):    self._training = False; return self

    def __setattr__(self, name, value):
        # Auto-register sub-modules and parameters
        object.__setattr__(self, name, value)
```

### 4.2 Core Layers

```python
# pinaka/nn/layers.py
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        # Kaiming uniform init (default in PyTorch)
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features) * scale,
                             requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Kaiming init
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
                             * np.sqrt(2/fan_in), requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        # Implement via im2col — convert convolution to matmul
        return self._conv2d_forward(x)

    def _im2col(self, x, kh, kw, stride, padding):
        """Unfold input patches into columns for matmul-based convolution."""
        ...
```

### 4.3 Normalization

```python
class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma   = Tensor(np.ones(num_features), requires_grad=True)   # scale
        self.beta    = Tensor(np.zeros(num_features), requires_grad=True)  # shift
        self.eps     = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)

    def forward(self, x):
        if self._training:
            mu  = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
            self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mu, var = self.running_mean, self.running_var
        x_norm = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 4.4 Attention & Transformer

```python
# pinaka/nn/attention.py
class ScaledDotProductAttention(Module):
    """Attention(Q, K, V) = softmax(QKᵀ / √d_k) V"""
    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask * -1e9
        weights = softmax(scores, dim=-1)
        return weights @ V, weights

class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, T, D = Q.shape
        def split_heads(x):
            return x.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        Q, K, V = split_heads(self.W_q(Q)), split_heads(self.W_k(K)), split_heads(self.W_v(V))
        out, _ = ScaledDotProductAttention()(Q, K, V, mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.W_o(out)

class TransformerEncoderLayer(Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ff1   = Linear(d_model, d_ff)
        self.ff2   = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop  = Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm formulation (more stable than original post-norm)
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.drop(self.ff2(relu(self.ff1(self.norm2(x)))))
        return x
```

### 4.5 Optimizers

```python
# pinaka/optim/adam.py
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]   # 1st moment
        self.v = [np.zeros_like(p.data) for p in self.params]   # 2nd moment

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad.data + self.weight_decay * p.data   # AdamW: decay on params, not grad
            self.m[i] = b1 * self.m[i] + (1-b1) * g
            self.v[i] = b2 * self.v[i] + (1-b2) * g**2
            m_hat = self.m[i] / (1 - b1**self.t)           # bias correction
            v_hat = self.v[i] / (1 - b2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
```

### 4.6 Training Loop (canonical pattern)

```python
def train(model, dataloader, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}")
```

### 4.7 Training Milestones

**Milestone 1 — MNIST MLP:**
```python
model = Sequential(
    Linear(784, 256), ReLU(),
    Linear(256, 128), ReLU(),
    Dropout(0.2),
    Linear(128, 10)
)
# Target: >96% test accuracy in 5 epochs
```

**Milestone 2 — CIFAR-10 CNN:**
```python
model = Sequential(
    Conv2d(3, 32, 3, padding=1), BatchNorm2d(32), ReLU(),
    Conv2d(32, 64, 3, padding=1), BatchNorm2d(64), ReLU(), MaxPool2d(2),
    # ...
    Linear(64*8*8, 10)
)
# Target: >75% test accuracy
```

**Milestone 3 — Character-level language model:**
```python
model = TransformerLM(vocab_size=65, d_model=256, num_heads=4,
                      num_layers=4, max_seq_len=256)
# Train on Shakespeare. Target: coherent text generation.
```

---

## Phase 5 — ONNX Export & FastAPI Serving

**Goal:** Turn trained pinaka models into portable, production-ready APIs.

**What you learn:** How model graphs are serialized. How inference serving works. What makes a model endpoint production-grade.

### 5.1 ONNX Export

ONNX represents a computation graph as a protobuf. Your autograd graph is already a computation graph — exporting is a traversal that maps pinaka ops to ONNX op types:

```python
# pinaka/export/onnx_export.py
import onnx
from onnx import helper, TensorProto

PINAKA_TO_ONNX = {
    'MatMul':  'MatMul',
    'Add':     'Add',
    'Relu':    'Relu',
    'Sigmoid': 'Sigmoid',
    'Reshape': 'Reshape',
    'Conv2d':  'Conv',
    'BatchNorm': 'BatchNormalization',
}

def export(model, dummy_input, path, opset=17):
    """Trace the model with dummy_input and export computation graph to ONNX."""
    # Run forward pass to build the graph
    with pinaka.autograd.record():
        output = model(dummy_input)

    nodes = []
    for node in _topological_sort(output):
        if node._grad_fn is None: continue
        op_type = PINAKA_TO_ONNX[node._grad_fn.__class__.__name__]
        nodes.append(helper.make_node(
            op_type,
            inputs=[str(id(inp)) for inp in node._inputs],
            outputs=[str(id(node))]
        ))

    graph = helper.make_graph(nodes, 'pinaka_model',
        inputs=[helper.make_tensor_value_info(str(id(dummy_input)), TensorProto.FLOAT, dummy_input.shape)],
        outputs=[helper.make_tensor_value_info(str(id(output)), TensorProto.FLOAT, output.shape)])

    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])
    onnx.save(model_proto, path)
```

**Validation:**
```python
import onnxruntime as ort
def validate_onnx(pinaka_model, onnx_path, test_input):
    pinaka_out = pinaka_model(test_input).data
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {'input': test_input.data})[0]
    assert np.allclose(pinaka_out, onnx_out, atol=1e-5), "ONNX export mismatch"
```

### 5.2 FastAPI Model Server

```python
# pinaka/serve/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

class ModelServer:
    def __init__(self, model, input_schema=None, output_schema=None):
        self.model = model
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self):
        @self.app.get("/health")
        def health():
            return {"status": "ok", "model": self.model.__class__.__name__}

        @self.app.get("/metadata")
        def metadata():
            return {"parameters": sum(p.data.size for p in self.model.parameters())}

        @self.app.post("/predict")
        def predict(data: dict):
            try:
                x = Tensor(np.array(data["input"]))
                with no_grad():
                    out = self.model(x)
                return {"prediction": out.data.tolist()}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

    def serve(self, host="0.0.0.0", port=8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

# Usage:
# server = ModelServer(trained_mnist_model)
# server.serve()
# curl -X POST http://localhost:8000/predict -d '{"input": [[...pixel values...]]}'
```

### 5.3 Model Registry

```python
# pinaka/serve/registry.py
class ModelRegistry:
    def __init__(self, storage_path="./models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self._active = {}

    def register(self, name, model, version=None):
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.storage_path / name / version
        path.mkdir(parents=True, exist_ok=True)
        pinaka.save(model, path / "model.pkl")
        metadata = {"name": name, "version": version,
                    "params": sum(p.data.size for p in model.parameters())}
        json.dump(metadata, open(path / "metadata.json", "w"))
        return version

    def load(self, name, version="latest"):
        if version == "latest":
            version = sorted((self.storage_path / name).iterdir())[-1].name
        return pinaka.load(self.storage_path / name / version / "model.pkl")

    def set_active(self, name, version):
        self._active[name] = self.load(name, version)
```

**Milestone:** Train MNIST MLP → export to ONNX → validate round-trip → serve via FastAPI → hit `/predict` with raw pixels, get digit back in <10ms. Model registry stores 3 versions and hot-swaps between them.

---

## Phase 6 — C++ Backend & CUDA

**Goal:** Replace the NumPy backend with a C++ CPU implementation and CUDA GPU kernels. The existing test suite should pass unchanged — the backend swap is transparent.

**What you learn:** Memory layout, cache efficiency, parallelism, GPU programming model.

**Pre-requisite:** Run `pytest` against the NumPy backend. Everything green? You're ready.

### 6.1 pybind11 Setup

```toml
# pyproject.toml additions
[build-system]
requires = ["poetry-core", "pybind11"]
```

```cpp
// src/ops.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

py::array_t<float> matmul_cpu(
    py::array_t<float> a,
    py::array_t<float> b
) {
    auto buf_a = a.request(), buf_b = b.request();
    int M = buf_a.shape[0], K = buf_a.shape[1], N = buf_b.shape[1];
    auto result = py::array_t<float>({M, N});
    auto buf_r  = result.request();

    float *pa = (float*)buf_a.ptr;
    float *pb = (float*)buf_b.ptr;
    float *pr = (float*)buf_r.ptr;

    // Tiled matmul for cache efficiency
    const int TILE = 32;
    for (int i = 0; i < M; i += TILE)
      for (int j = 0; j < N; j += TILE)
        for (int k = 0; k < K; k += TILE)
          for (int ii = i; ii < std::min(i+TILE, M); ii++)
            for (int kk = k; kk < std::min(k+TILE, K); kk++)
              for (int jj = j; jj < std::min(j+TILE, N); jj++)
                pr[ii*N+jj] += pa[ii*K+kk] * pb[kk*N+jj];

    return result;
}

PYBIND11_MODULE(pinaka_cpp, m) {
    m.def("matmul", &matmul_cpu, "Tiled CPU matmul");
}
```

```python
# pinaka/core/backend_cpp.py
import pinaka_cpp

class CppBackend:
    def matmul(self, a, b):
        return pinaka_cpp.matmul(a.astype(np.float32), b.astype(np.float32))
    # ... other ops
```

### 6.2 OpenMP Parallelism

```cpp
#include <omp.h>

// Parallel batch dimension
#pragma omp parallel for schedule(static)
for (int b = 0; b < batch_size; b++) {
    // process batch[b]
}
```

Add to `pyproject.toml`:
```toml
[tool.poetry.build-system]
extra-compile-args = ["-O3", "-fopenmp", "-march=native"]
```

### 6.3 CUDA Kernels

**Start here — tiled matrix multiply in CUDA. This is the canonical first CUDA exercise:**

```cuda
// src/kernels/matmul.cu
#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from global memory into shared memory
        sA[threadIdx.y][threadIdx.x] = (row < M && t*TILE_SIZE+threadIdx.x < K)
            ? A[row*K + t*TILE_SIZE+threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (t*TILE_SIZE+threadIdx.y < K && col < N)
            ? B[(t*TILE_SIZE+threadIdx.y)*N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row*N + col] = acc;
}
```

**Device management on the Tensor:**

```python
class Tensor:
    def to(self, device: str):
        if device == self.device: return self
        new_data = self._backend.to_device(self.data, device)
        t = Tensor(new_data, requires_grad=self.requires_grad, device=device)
        t.grad = self.grad
        return t

    def cuda(self): return self.to('cuda')
    def cpu(self):  return self.to('cpu')
```

### 6.4 Performance Benchmarks

After C++ and CUDA backends are working, benchmark:

| Operation | NumPy | C++ CPU | CUDA (expected) |
|---|---|---|---|
| Matmul 1024×1024 | baseline | ~3–5× faster | ~50–100× faster |
| Conv2d 3×224×224 batch=32 | baseline | ~2–4× faster | ~30–80× faster |
| MLP MNIST 1 epoch | baseline | ~2–3× faster | ~20–50× faster |

**Milestone:** MNIST MLP trains 20–50× faster on CUDA than NumPy. All existing tests (pinaka + numpy comparison) pass unchanged with `PINAKA_BACKEND=cuda`.

---

## Testing Strategy

Maintain three test layers throughout all phases:

### Layer 1: Correctness (vs NumPy ground truth)
```python
# Every op: assert np.allclose(pinaka_result, numpy_result)
# Tolerance: rtol=1e-5, atol=1e-8 for float64
```

### Layer 2: Gradient correctness (numerical grad check)
```python
# Every differentiable op: check_gradients(f, inputs, tol=1e-4)
```

### Layer 3: Behavioral (sklearn/PyTorch parity)
```python
# Classical ML: within 1% of sklearn on standard datasets
# Neural nets: same output as PyTorch given same weights and inputs
# ONNX: round-trip matches within 1e-5
```

### Suggested test structure
```
tests/
├── unit/
│   ├── test_tensor_ops.py       # Phase 1
│   ├── test_autograd.py         # Phase 2
│   ├── test_grad_check.py       # Phase 2
│   ├── test_models.py           # Phase 3
│   ├── test_nn_layers.py        # Phase 4
│   ├── test_optimizers.py       # Phase 4
│   └── test_onnx_roundtrip.py   # Phase 5
├── integration/
│   ├── test_mnist_mlp.py        # Phase 4 milestone
│   ├── test_cifar_cnn.py        # Phase 4 milestone
│   └── test_serve_api.py        # Phase 5 milestone
└── benchmarks/
    ├── bench_matmul.py           # Phase 6
    └── bench_training.py         # Phase 6
```

---

## Reading Schedule (aligned to phases)

### Phase 1 (Tensors)
- *Deep Learning from Scratch* Ch 1–2
- *Practical Statistics* Ch 1 (Exploratory Data Analysis — understand what you're computing)

### Phase 2 (Autograd)
- *Deep Learning from Scratch* Ch 4–5
- *ML from Scratch* — Introduction, Appendix (math review)

### Phase 3 (Classical ML)
Read in this order per algorithm:
1. *Practical Statistics* chapter on that family
2. *ML from Scratch* concept + construction sections
3. *Hands-On ML* chapter for breadth
4. **Implement in pinaka**

Schedule: Linear regression → Logistic → Naive Bayes → Decision Trees → Ensembles → Clustering → PCA

### Phase 4 (Deep Learning)
- *Deep Learning from Scratch* Ch 6–end (all of it)
- *Hands-On ML* Part 2 Ch 10–16 (parallel reading)

### Phase 5 (Serving)
- ONNX spec: https://onnx.ai/onnx/intro/
- FastAPI docs: https://fastapi.tiangolo.com

### Phase 6 (C++/CUDA)
- NVIDIA CUDA Programming Guide (Chapter 1–5)
- "Programming Massively Parallel Processors" (Kirk & Hwu) — optional deep dive

---

## Implementation Timeline (rough)

| Phase | Effort estimate | Depends on |
|---|---|---|
| 1 — Tensor engine | 2–3 weeks | Nothing |
| 2 — Autograd | 3–4 weeks | Phase 1 |
| 3 — Stats & Classical ML | 6–8 weeks | Phase 2 (optional — can use numpy directly) |
| 4 — Deep Learning | 6–8 weeks | Phase 2 |
| 5 — ONNX & Serving | 2–3 weeks | Phase 4 |
| 6 — C++ & CUDA | 4–8 weeks | Phase 1, 2 |
| **Total** | **~6–8 months** | |

---

## Key Design Principles

1. **Backend abstraction first.** Every numerical op goes through `self._backend`. This is the single most important design decision in the entire project.

2. **Test against ground truth always.** NumPy for tensors. PyTorch for neural nets. sklearn for classical ML. ONNX Runtime for exports. Never trust a new op without a ground-truth comparison.

3. **Statistics and ML share one foundation.** Every loss function is a negative log-likelihood. Derive each loss from MLE before implementing it — this prevents the "magic numbers" feeling.

4. **Numerical gradient check is your oracle.** If backward() passes the numerical check, it is correct. This is the most valuable single function in the codebase.

5. **Build vertically before horizontally.** Train MNIST before implementing LSTM. Implement LinearRegression fully (including tests, gradient check, sklearn parity) before starting DecisionTree. Depth first.
