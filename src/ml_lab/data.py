"""Local, reproducible data sources; no remote downloads required."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def regression_data(n=400, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y_true = 6*x**4 - 2*x**3 - 3*x**2 + 4*x
    y = y_true + rng.normal(0, 0.3, n)
    x_train, x_test, y_train, y_test = train_test_split(x[:, None], y, test_size=.2, random_state=seed)
    return x_train, x_test, y_train, y_test


def digits_data(seed=42):
    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=.2, stratify=data.target, random_state=seed)
    return X_train, X_test, y_train, y_test


def scaled_digits(seed=42):
    X_train, X_test, y_train, y_test = digits_data(seed)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test, scaler
