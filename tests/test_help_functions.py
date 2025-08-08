import numpy as np

from hdmpy import cvec, cor, init_values


def test_cvec_shapes():
    a = [1, 2, 3]
    v = cvec(a)
    assert v.shape == (3, 1)

    b = np.array([[1, 2, 3]])  # row vector
    v2 = cvec(b)
    assert v2.shape == (3, 1)

    c = np.array([[1], [2]])  # already column
    v3 = cvec(c)
    assert v3.shape == (2, 1)


def test_cor_simple():
    # y perfectly correlated with first column, anti with second
    y = cvec([1, 2, 3, 4])
    X = np.column_stack(([1, 2, 3, 4], [4, 3, 2, 1]))
    r = cor(y, X)
    assert r.shape == (2,)
    assert np.isclose(r[0], 1.0)
    assert np.isclose(r[1], -1.0)


def test_init_values_shapes_and_non_nan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    beta = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ beta + rng.normal(size=50) * 0.1
    res = init_values(X, y, number=3, intercept=True)
    assert 'residuals' in res and 'coefficients' in res
    assert res['coefficients'].shape == (5, 1)
    assert res['residuals'].shape == (50, 1)
    assert np.all(np.isfinite(res['coefficients']))

