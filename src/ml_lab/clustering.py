"""Gaussian mixtures: both from-scratch EM and sklearn reference."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


class GaussianMixtureEM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-5, seed=42):
        self.n_components, self.max_iter, self.tol = n_components, max_iter, tol
        self.reg_covar, self.seed = reg_covar, seed

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if self.n_components > n or self.n_components < 1:
            raise ValueError('invalid number of components')
        rng = np.random.default_rng(self.seed)
        means = X[rng.choice(n, self.n_components, replace=False)].copy()
        cov = np.stack([np.cov(X, rowvar=False) + self.reg_covar*np.eye(d)]*self.n_components)
        weights = np.full(self.n_components, 1/self.n_components)
        self.history_ = []
        for _ in range(self.max_iter):
            log_joint = np.column_stack([
                np.log(max(weights[k], 1e-15)) + multivariate_normal.logpdf(
                    X, means[k], cov[k], allow_singular=False)
                for k in range(self.n_components)])
            log_norm = logsumexp(log_joint, axis=1)
            ll = float(log_norm.sum())
            self.history_.append(ll)
            responsibilities = np.exp(log_joint - log_norm[:, None])
            counts = np.maximum(responsibilities.sum(axis=0), 1e-12)
            weights = counts/n
            means = responsibilities.T @ X / counts[:, None]
            for k in range(self.n_components):
                diff = X-means[k]
                cov[k] = (responsibilities[:, k, None]*diff).T @ diff / counts[k]
                cov[k].flat[::d+1] += self.reg_covar
            if len(self.history_) > 1 and abs(ll-self.history_[-2]) < self.tol:
                break
        self.means_, self.covariances_, self.weights_ = means, cov, weights
        self.n_iter_ = len(self.history_)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_joint = np.column_stack([
            np.log(max(self.weights_[k], 1e-15)) + multivariate_normal.logpdf(
                X, self.means_[k], self.covariances_[k])
            for k in range(self.n_components)])
        return np.exp(log_joint-logsumexp(log_joint, axis=1, keepdims=True))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
