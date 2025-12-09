# Experiments summary

We train models to predict the percentage return on investment (`ret_btc_next`), rather than the specific Bitcoin price. The goal is to determine the percentage gain or loss if we buy today and sell tomorrow.

## Linear regression

The code is located in `models/linear_regression_train.py`.  
We use the same features as in the random forest model and the same target `ret_btc_next` (next-day BTC return). The data is sorted by `open_time`, and we do a chronological split into train and test (`test_size = 0.2`, no shuffling).

We trained three linear models:

- **OLS** – plain linear regression,
- **Ridge** – L2-regularized regression,
- **Lasso** – L1-regularized regression (with a `StandardScaler` in a sklearn `Pipeline`).

For Lasso we tuned the regularization strength α on a grid  
`[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]`  
on our dataset (2088 rows → 1670 train / 418 test). The best result was obtained for **α = 0.005**, which we use as the final linear model.

**Best Lasso model (α = 0.005):**

- MAE: **0.01674**
- RMSE: **0.02322**

**Naive baseline** (always predict `ret_btc_next = 0`):

- MAE: **0.02431**
- RMSE: **0.04379**

Thus, although L1-regularized linear regression improves over the naive strategy in terms of MAE and RMSE, inspection of the Lasso coefficients shows that most feature weights are shrunk to zero. This suggests that the model does not exploit strong linear relationships in the features, but instead effectively learns a slightly better constant prediction (non-zero average return) than the naive assumption `ret_btc_next = 0`.

## Random forest

The code is located in `models/random_forest_train.py`.
We determined that the following parameters would be most important for our model:

- max_depth
- min_samples_leaf
- max_features
- n_estimators
- max_samples
- criterion

We also experimented with disabling specific features (e.g., `open_btc`, `volume_btc`, `ntrades_eth`...).

We tested these ranges using Grid Search:

```python
    GRID_MAX_DEPTH = [3, 5, 7, 9, 11, 13, 15]
    GRID_MIN_SAMPLES_LEAF = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    GRID_MAX_FEATURES = ["sqrt", "log2", 0.7, 1.0]
    GRID_N_ESTIMATORS = [50, 100, 200, 300]
    GRID_MAX_SAMPLES = [0.3, 0.5, 0.7]
```

### Experiment Results

The best model found:

```bash
=== BEST MODEL FROM GRID SEARCH ===
Best hyperparameters:
  max_depth: 3
  min_samples_leaf: 300
  max_features: log2
  n_estimators: 50
  max_samples: 0.5
Metrics on test set:
  mae: 0.016587117185736317
  rmse: 0.023164909256702313
  n_train: 1670
  n_test: 418


Naive baseline (ret_tomorrow = 0):
  mae: 0.024311851021388362
  rmse: 0.04379106572596224
```

Random Forest outperformed the naive baseline (_MAE 0.016 vs 0.024_), but the selected parameters make us concerned:

- **Shallow trees (`max_depth: 1`)**: The trees make only one split. This means the model isn't connecting facts (it doesn't see dependencies between features) and applies only the simplest possible rules.
- **Very large groups (`min_samples_leaf: 300`)**: Each decision is based on an average from a very large portion of history. Consequently, the model heavily averages the predictions and may ignore fresh, short-term trends.
- **Few trees (`n_estimators: 50`)**: Adding more trees didn't improve results, which suggests there is a lot of noise in the data and it is difficult to extract concrete dependencies.
- **No advantage over regression**: We expected random forest to be significantly better but the results are close to standard linear regression. The forest didn't find any complex, non-linear patterns.
