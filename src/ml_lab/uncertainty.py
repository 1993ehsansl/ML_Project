"""Hoeffding's inequality for fixed classifiers chosen independently of evaluation data."""
import math


def hoeffding_epsilon(n, delta=.05):
    if n <= 0 or not 0 < delta < 1:
        raise ValueError('n must be positive and 0 < delta < 1')
    return math.sqrt(math.log(2/delta)/(2*n))


def empirical_risk_bound(errors, delta=.05):
    """Two-sided fixed-hypothesis bound; NOT a guarantee for test-set-tuned models."""
    n = len(errors)
    if n == 0:
        raise ValueError('empty evaluation set')
    risk = sum(bool(x) for x in errors)/n
    eps = hoeffding_epsilon(n, delta)
    return dict(empirical_risk=risk, epsilon=eps,
                lower=max(0., risk-eps), upper=min(1., risk+eps), n=n, delta=delta)
