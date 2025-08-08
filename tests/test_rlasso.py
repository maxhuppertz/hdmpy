import numpy as np

from hdmpy import rlasso


def test_rlasso_basic_flow_homoskedastic():
    rng = np.random.default_rng(2)
    n, p = 80, 10
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta = np.zeros((p, 1)); beta[0, 0] = 1.5
    y = (X @ beta).ravel() + rng.normal(size=n) * 0.1

    model = rlasso(X, y, post=True, homoskedastic=True, X_dependent_lambda=False,
                   numSim=500, numIter=5, par=False, fix_seed=True)

    est = model.est
    assert isinstance(est, dict)
    assert 'coefficients' in est and 'lambda' in est and 'residuals' in est
    # shape checks
    assert est['beta'].shape[0] == p
    assert est['residuals'].shape[0] == n

