import numpy as np

from hdmpy import rlassoEffect, rlassoEffects


def test_rlassoEffect_double_selection_runs():
    rng = np.random.default_rng(3)
    n, p = 60, 12
    X = rng.normal(size=(n, p))
    d = (X[:, 0] + rng.normal(size=n) * 0.1)
    y = (0.8 * d + rng.normal(size=n) * 0.2)

    res = rlassoEffect(X, y, d, method='double selection',
                       homoskedastic=True, X_dependent_lambda=False,
                       numSim=500, numIter=5, par=False, fix_seed=True)
    assert isinstance(res, dict)
    for k in ['alpha', 'se', 't', 'pval', 'samplesize']:
        assert k in res


def test_rlassoEffects_partialling_out_runs_small():
    rng = np.random.default_rng(4)
    n, p = 50, 6
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.1

    fx = rlassoEffects(X, y, method='partialling out', par_outer=False,
                       par_inner=False, par_any=False, numSim=200, numIter=4,
                       fix_seed=True)
    # ensure basic attributes are populated
    assert hasattr(fx, 'res') and isinstance(fx.res, dict)
    assert 'coefficients' in fx.res and 'se' in fx.res

