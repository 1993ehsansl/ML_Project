import numpy as np
from ml_lab.bayesian import BayesianPolynomialRegression, select_degree
from ml_lab.clustering import GaussianMixtureEM
from ml_lab.uncertainty import hoeffding_epsilon, empirical_risk_bound
from ml_lab.neural import DigitsCNN, RegressionMLP


def test_bayesian_fit_and_uncertainty():
    x = np.linspace(-1, 1, 50)[:, None]
    y = x[:, 0]**2
    model = BayesianPolynomialRegression(2, noise_var=.01).fit(x, y)
    pred, std = model.predict(x, return_std=True)
    assert np.mean((pred-y)**2) < .01
    assert np.all(std > 0)
    assert np.isfinite(model.log_marginal_likelihood(x, y))
    degree, scores = select_degree(x, y, range(1, 4), .01)
    assert degree in scores


def test_hoeffding():
    assert hoeffding_epsilon(1000) < hoeffding_epsilon(100)
    bound = empirical_risk_bound([0, 0, 1, 0])
    assert bound['lower'] <= .25 <= bound['upper']


def test_gmm_em():
    rng = np.random.default_rng(4)
    x = np.vstack([rng.normal(-3, .4, (50, 2)), rng.normal(3, .4, (50, 2))])
    model = GaussianMixtureEM(2, seed=4).fit(x)
    probs = model.predict_proba(x)
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.all(np.isfinite(model.history_))
    assert model.history_[-1] >= model.history_[0] - 1e-4


def test_neural_shapes():
    import torch
    assert RegressionMLP()(torch.zeros(2, 1)).shape == (2, 1)
    assert DigitsCNN()(torch.zeros(2, 1, 8, 8)).shape == (2, 10)
