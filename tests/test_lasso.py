import numpy as np

from hdmpy import LassoShooting_fit


def test_LassoShooting_fit_runs_and_shapes():
    rng = np.random.default_rng(1)
    n, p = 60, 8
    X = rng.normal(size=(n, p)).astype(float)
    true_beta = np.zeros((p, 1))
    true_beta[0, 0] = 1.0
    y = (X @ true_beta).ravel() + rng.normal(size=n) * 0.05

    # variable-specific penalty terms
    lmbda = np.ones((p, 1)) * 0.1

    out = LassoShooting_fit(X, y, lmbda, maxIter=200)

    assert set(['coefficients', 'coef.list', 'num.it']).issubset(out.keys())
    w = out['coefficients']
    wp = out['coef.list']
    assert w.shape == (p, 1)
    assert wp.shape[0] == p and wp.shape[1] >= 1

