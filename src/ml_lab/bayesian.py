"""Closed-form Bayesian polynomial regression with Gaussian prior/noise."""
import numpy as np
from scipy.special import logsumexp


class BayesianPolynomialRegression:
    def __init__(self, degree=4, noise_var=.09, prior_var=25.):
        if degree < 0 or noise_var <= 0 or prior_var <= 0:
            raise ValueError("degree must be >=0 and variances must be positive")
        self.degree, self.noise_var, self.prior_var = degree, noise_var, prior_var

    def features(self, X):
        x = np.asarray(X).reshape(-1)
        return np.vander(x, N=self.degree + 1, increasing=True)

    def fit(self, X, y):
        Phi = self.features(X)
        y = np.asarray(y).reshape(-1)
        precision = (Phi.T @ Phi)/self.noise_var + np.eye(Phi.shape[1])/self.prior_var
        self.cov_ = np.linalg.solve(precision, np.eye(Phi.shape[1]))
        self.mean_ = np.linalg.solve(precision, Phi.T @ y/self.noise_var)
        return self

    def predict(self, X, return_std=False):
        Phi = self.features(X)
        mean = Phi @ self.mean_
        if not return_std:
            return mean
        # Includes aleatoric observation noise AND epistemic parameter uncertainty.
        variance = self.noise_var + np.einsum('ij,jk,ik->i', Phi, self.cov_, Phi)
        return mean, np.sqrt(np.maximum(variance, 0))

    def log_marginal_likelihood(self, X, y):
        Phi, yy = self.features(X), np.asarray(y).ravel()
        K = self.prior_var * Phi @ Phi.T + self.noise_var*np.eye(len(yy))
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            raise ValueError("Covariance is not positive definite")
        return -.5*(yy @ np.linalg.solve(K, yy) + logdet + len(yy)*np.log(2*np.pi))


def select_degree(X, y, degrees=range(1, 9), noise_var=.09):
    """Select complexity by BIC on training data (not test data)."""
    from sklearn.metrics import mean_squared_error
    best = None
    scores = {}
    n = len(y)
    for degree in degrees:
        model = BayesianPolynomialRegression(degree, noise_var=noise_var).fit(X, y)
        mse = max(mean_squared_error(y, model.predict(X)), 1e-12)
        bic = n*np.log(mse) + (degree+1)*np.log(n)
        scores[degree] = float(bic)
        if best is None or bic < scores[best]:
            best = degree
    return best, scores
