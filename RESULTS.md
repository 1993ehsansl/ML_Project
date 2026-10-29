# Reproduced local benchmark results

Run: `PYTHONPATH=src python -m ml_lab.cli --task all --seed 42` (full configuration, not `--quick`).

| Task | Model | Metric | Observed value |
|---|---|---|---:|
| Classification (scikit-learn Digits, 360 holdout) | SVM RBF | Accuracy | 98.06% |
| Classification | Logistic regression | Accuracy | 97.22% |
| Classification | PyTorch CNN | Accuracy | 94.44% |
| Regression (synthetic quartic) | Bayesian polynomial | RMSE | 0.3162 |
| Regression | SVR | RMSE | 0.3311 |
| Regression | PyTorch MLP | RMSE | 0.4079 |
| Regression | Bayesian polynomial | Nominal 95% interval coverage | 95.0% |
| Clustering (2D PCA digit features, 360 holdout) | Custom GMM EM | ARI | 0.2371 |
| Clustering | sklearn GMM | ARI | 0.2807 |

All results come from a single seeded split; no repeated runs, confidence intervals over seeds, or external validation have been completed. In particular, the clustering ARI is modest and these digits results must not be presented as MNIST performance. The clustering algorithm itself is fit on the held-out unlabeled set (transductive experiment). The fixed-classifier Hoeffding bound upper end for SVM error is 9.10% at delta=0.05; it is conservative.
