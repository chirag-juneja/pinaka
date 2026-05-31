# Pinaka Phase 1 Roadmap: Stats, Clean, & Classical ML

A hands-on engineering roadmap to build data preprocessing pipelines, descriptive statistics engines, and analytical classical models driven directly by messy, real-world tabular data.

---

## Phase 1: Messy Synthetic Data Infrastructure (`pinaka.stats.sampling`)
*Objective: Build an offline sandbox that simulates data corruption (null values, string units, categorical strings) so you can test your pipeline algorithms deterministically.*

- [ ] **Create `make_messy_tabular_data()` utility:**
  - Generate a dataset containing a clean target array $y$, but a feature matrix $X$ corrupted with:
    - Randomly injected `NaN` values.
    - Categorical columns (e.g., `["Manual", "Automatic"]`).
    - Numeric values masked as strings (e.g., `["1197 CC", "999 CC"]`).
    - Skewed outliers to test robust scaling.

---

## Phase 2: Descriptive Statistics Core (`pinaka.stats.descriptive`)
*Objective: Build the mathematical foundation required for data cleaning thresholds and modeling.*

- [ ] **Central Tendency & Dispersion Engine:**
  - Implement `mean(x)`, `median(x)` (sorting arrays to find the middle boundary without `np.median`), and `mode(x)`.
  - Implement variance (`var`) and standard deviation (`std`).
- [ ] **Matrix Association Utilities:**
  - Implement covariance matrices and Pearson Correlation Coefficient (`pearson_r`) from scratch to allow feature correlation analysis directly inside Pinaka.
- [ ] **Quantile & Outlier Identifiers:**
  - Implement a raw percentile/quantile parser to calculate the Interquartile Range ($IQR = Q3 - Q1$).

---

## Phase 3: Stateful Preprocessing & Cleanup (`pinaka.ml.preprocessing`)
*Objective: Build an object-oriented preprocessing API. Classes must use a `.fit()` method to learn parameters from training data and a `.transform()` method to apply them.*

- [ ] **The Imputer Module (`SimpleImputer`):**
  - Create a class that fills `NaN` elements using your stats module's `mean`, `median`, or a constant `mode`.
- [ ] **The Numerical Parsers:**
  - Create a custom regex or string-splitting utility to strip units (like `"CC"`, `"kmpl"`, `"bhp"`) from data blocks and convert strings seamlessly back into floating-point vectors.
- [ ] **Categorical Encoders:**
  - `OneHotEncoder`: Convert low-cardinality categorical strings into a matrix of binary bits ($0$ or $1$).
  - `LabelEncoder`: Map ordered strings to unique integers.
- [ ] **Feature Scalers:**
  - `StandardScaler`: $X_{scaled} = \frac{X - \mu}{\sigma}$ (using internal running states for $\mu$ and $\sigma$).
  - `MinMaxScaler`: Scale attributes precisely between $[0, 1]$.

---

## Phase 4: Feature Engineering Utilities (`pinaka.ml.feature_engineering`)
*Objective: Maximize the predictive capacity of linear and logistic models by modifying existing dimensions.*

- [ ] **Polynomial Features Intersector (`PolynomialFeatures`):**
  - Implement an engine that maps degree configurations to inputs (e.g., transforming features $[a, b]$ into $[1, a, b, a^2, ab, b^2]$) to capture non-linear trends.
- [ ] **Outlier Clipper (`RobustClipper`):**
  - Use your $IQR$ statistics calculator to clip values falling outside lower and upper bounds ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$).

---

## Phase 5: Classical Optimization Models (`pinaka.ml.linear_model`)
*Objective: Consume the cleaned, engineered feature pipelines into classical models.*

- [ ] **Split Validation Utility:**
  - Build a vectorized `train_test_split(X, y, test_size, shuffle=True)` function.
- [ ] **Ordinary Least Squares Regression (`LinearRegression`):**
  - Add a vector of ones to handle intercept weights: $X_{bias} = [1, X]$.
  - Solved analytically: $\theta = (X^T X)^{-1} X^T y$.
- [ ] **Logistic Regression Class (`LogisticRegression`):**
  - Implement the sigmoid mapping vector function: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  - Train iteratively via Gradient Descent utilizing a binary cross-entropy cost matrix loop.

---

## Phase 6: Validation Sandboxes (`/examples`)
*Objective: Verify Pinaka's performance from parsing raw files to computing performance metrics.*

- [ ] **`01_car_prices_pipeline.py` script:**
  - Load a messy car details CSV.
  - Pipe text values into your framework string-parsers.
  - Handle null values with `SimpleImputer`.
  - Transform categorical variables with `OneHotEncoder`.
  - Engineering: Derive an `age` feature from the manufacturing year.
  - Split data, train `LinearRegression`, and output evaluation scores ($R^2$ and $MSE$).
