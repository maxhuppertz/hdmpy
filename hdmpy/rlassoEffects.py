################################################################################
### Python port of rlassoEffects.R
### https://github.com/cran/hdm/blob/master/R/rlassoEffects.R
################################################################################

################################################################################
### 1: Load modules
################################################################################

# Standard Python modules
import joblib as jbl
import multiprocess as mp
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import multivariate_normal, norm
from sklearn.linear_model import LinearRegression as lm

# Other parts of hmdpy
from hdmpy.help_functions import cvec
from hdmpy.rlasso import rlasso

################################################################################
### 2: Define functions
################################################################################

################################################################################
### 2.1: Functions which are not in the original R package
###      These are generally helper functions to allow an implementation which
###      reads as closely to the original R code as possible, and to ease a
###      Python implementation, including parallelizing the code
################################################################################


# Define a function which calculates the homoskedastic variance estimator for
# OLS, based on X and the residuals
def get_cov(X, e, add_intercept=True, homoskedastic=False):
    """ Calculates OLS variance estimator based on X and residuals

    Inputs
    X: n by k matrix, RHS variables
    e: n by 1 vector or vector-like, residuals from an OLS regression
    add_intercept: Boolean, if True, adds an intercept as the first column of X
                   (and increases k by one)

    Outputs
    V_hat: k by k NumPy array, estimated covariance matrix
    """
    # Get the number of observations n and parameters k
    n, k = X.shape

    # Check whether an intercept needs to be added
    if add_intercept:
        # If so, add the intercept
        X = np.concatenate([np.ones(shape=(n,1)), X], axis=1)

        # Don't forget to increase k
        k = k + 1

    # Make sure the residuals are a proper column vector
    e = cvec(e)

    # Calculate X'X
    XX = X.T @ X

    # Calculate its inverse
    XXinv = linalg.inv(XX)

    # Check whether to use homoskedastic errors
    if homoskedastic:
        # If so, calculate the homoskedastic variance estimator
        V_hat = (1 / (n-k)) * XXinv * (e.T @ e)
    else:
        # Otherwise, calculate an intermediate object
        S = (e @ np.ones(shape=(1,k))) * X

        # Then, get the HC0 sandwich estimator
        V_hat = (n / (n-k)) * XXinv @ (S.transpose() @ S) @ XXinv

    # Return the result
    return V_hat


# Define a function which wraps rlassoEffect, so it can easily be parallelized
# within rlassoEffects()
def rlassoEffect_wrapper(i, x, y, d, method='double selection', I3=None,
                         post=True, colnames_d=None, colnames_x=None,
                         intercept=True, model=True, homoskedastic=False,
                         X_dependent_lambda=False, lambda_start=None, c=1.1,
                         gamma=None, numSim=5000, numIter=15, tol=10**(-5),
                         threshold=-np.inf, par=True, corecap=np.inf,
                         fix_seed=True, verbose=False):
    """ Wrapper for rlassoEffect()

    Inputs
    i: Integer, index of the current variable of interest

    See the rlassoEffect() documentation for other inputs

    Output
    res: Dictionary, contains a collection of results from rlassoEffect(), or a
         collection of empty strings and NANs if an error is encountered while
         running rlassoEffect()
    """
    if np.amin(x.shape) == 1:
        x = cvec(x)

    y = cvec(y)

    d = cvec(d)

    try:
        col = rlassoEffect(x, y, d, method=method, I3=I3, post=post,
                           colnames_d=colnames_d, colnames_x=colnames_x,
                           intercept=intercept, model=model,
                           homoskedastic=homoskedastic,
                           X_dependent_lambda=X_dependent_lambda,
                           lambda_start=lambda_start, c=c, gamma=gamma,
                           numSim=numSim, numIter=numIter, tol=tol,
                           threshold=threshold, par=par, corecap=corecap,
                           fix_seed=fix_seed)

        smat = np.zeros(shape=(x.shape[1]+1, 1)) * np.nan
        smat[np.arange(smat.shape[0]) != i] = col['selection_index']

        res = {
            'coefficients': [i, col['alpha']],
            'se': [i, col['se'][0]],
            't': [i, col['t'][0]],
            'pval': [i, col['pval'][0]],
            'lasso_regs': {i: col},
            'reside': [i, col['residuals']['epsilon']],
            'residv': [i, col['residuals']['v']],
            'coef_mat': {i: col['coefficients_reg']},
            'selection_matrix': [i, smat]
        }
    except Exception as e:
        # Mimic the results in the original code, where any errors result in a
        # variable being skipped, and the preallocated results arrays containing
        # either NANs or empty lists
        res = {
            'coefficients': [i, np.nan],
            'se': [i, np.nan],
            't': [i, np.nan],
            'lasso_regs': {i: e},
            'pval': [i, np.nan],
            'reside': [i, np.zeros(shape=(x.shape[0], 1)) * np.nan],
            'residv': [i, np.zeros(shape=(x.shape[0], 1)) * np.nan],
            'coef_mat': {i: []},
            'selection_matrix': [i, np.zeros(shape=(x.shape[1]+1, 1)) * np.nan]
        }

        if verbose:
            print('Error encountered in rlassoEffect_wrapper()')
            print(e)
            print()

    return res


# Define a function to simulate quantiles needed for joint confidence intervals
def simul_ci(k=1, Omega=None, var=None, seed=0, fix_seed=True, verbose=False):
    if Omega is None:
        Omega = np.identity(k)
    else:
        k = Omega.shape[0]

    if var is None:
        var = np.diag(Omega)

    try:
        if fix_seed:
            # This is a key difference between the R and Python implementation.
            # For some data sets, especially when k > n, scipy.stats.norm() will
            # return an error, claiming than Omega is singular. R's
            # MASS::mvrnorm(), on the other hand, will happily use Omega and
            # calculate draws from it. I had to add allow_singular to get both
            # implementations to work similarly.
            beta = multivariate_normal(cov=Omega, allow_singular=True).rvs(
                random_state=seed
            )
        else:
            beta = multivariate_normal(cov=Omega, allow_singular=True).rvs()

        sim = np.amax(np.abs(cvec(beta) / cvec(np.sqrt(var))))
    except Exception as e:
        if verbose:
            print('Error encountered in simul_ci():')
            print(e)
            print()

        sim = np.nan

    return sim

################################################################################
### 2.2: Functions which are in the original R package
################################################################################


def rlassoEffect(x, y, d, method='double selection', I3=None, post=True,
                 colnames_d=None, colnames_x=None, intercept=True,
                 model=True, homoskedastic=False, X_dependent_lambda=False,
                 lambda_start=None, c=1.1, gamma=None, numSim=5000, numIter=15,
                 tol=10**(-5), threshold=-np.inf, par=True, corecap=np.inf,
                 fix_seed=True):
    d = cvec(d)

    y = cvec(y)

    n, kx = x.shape

    if colnames_d is None:
        colnames_d = ['d1']

    if (colnames_x is None) and (x is not None):
        colnames_x = ['x' + str(i) for i in np.arange(kx)]

    if method == 'double selection':
        I1 = rlasso(x, d, post=post, colnames=colnames_x, intercept=intercept,
                    model=model, homoskedastic=homoskedastic,
                    X_dependent_lambda=X_dependent_lambda,
                    lambda_start=lambda_start, c=c, gamma=gamma, numSim=numSim,
                    numIter=numIter, tol=tol, threshold=threshold, par=par,
                    corecap=corecap, fix_seed=fix_seed).est['index']
        I2 = rlasso(x, y, post=post, colnames=colnames_x, intercept=intercept,
                    model=model, homoskedastic=homoskedastic,
                    X_dependent_lambda=X_dependent_lambda,
                    lambda_start=lambda_start, c=c, gamma=gamma, numSim=numSim,
                    numIter=numIter, tol=tol, threshold=threshold, par=par,
                    corecap=corecap, fix_seed=fix_seed).est['index']

        # Original code checks if type(I3) is bool, but I believe they only do
        # that to see whether it has been defined by the user
        if I3 is not None:
            I3 = cvec(I3)

            I = cvec(I1.astype(bool) | I2.astype(bool) | I3.astype(bool))
        else:
            I = cvec(I1.astype(bool) | I2.astype(bool))
            # missing here: names(I) <- union(names(I1),names(I2))

        if I.sum() == 0:
            I = None

        x = np.concatenate([d, x[:, I[:,0]]], axis=1)

        reg1 = lm(fit_intercept=True).fit(x, y)

        alpha = reg1.coef_[0,0]

        names_alpha = colnames_d

        resid = y - cvec(reg1.predict(x))

        if I is None:
            xi = (resid) * np.sqrt(n/(n - 1))
        else:
            xi = (resid) * np.sqrt(n/(n - I.sum() - 1))

        if I is None:
            # Fit an intercept-only model
            reg2 = lm(fit_intercept=False).fit(np.ones_like(d), d)

            v = d - cvec(reg2.predict(np.ones_like(d)))
        else:
            reg2 = lm(fit_intercept=True).fit(x[:, 1:], d)

            v = d - cvec(reg2.predict(x[:, 1:]))

        var = (
            (1/n)
            * (1/np.mean(v**2, axis=0))
            * np.mean((v**2) * (xi**2), axis=0)
            * (1/np.mean(v**2, axis=0))
        )

        se = np.sqrt(var)

        tval = alpha / np.sqrt(var)

        pval = 2 * norm.cdf(-np.abs(tval))

        if I is None:
            no_selected = 1
        else:
            no_selected = 0

        res = {'epsilon': xi, 'v': v}

        if np.issubdtype(type(colnames_d), np.str_):
            colnames_d = [colnames_d]

        results = {
            'alpha': alpha,
            #'se': pd.DataFrame(se, index=colnames_d),
            'se': se,
            't': tval,
            'pval': pval,
            'no_selected': no_selected,
            'coefficients': alpha,
            'coefficient': alpha,
            'coefficients_reg': reg1.coef_,
            'selection_index': I,
            'residuals': res,
            #call = match.call(),
            'samplesize': n
        }
    elif method == 'partialling out':
        reg1 = rlasso(x, y, post=post, colnames=colnames_x, intercept=intercept,
                      model=model, homoskedastic=homoskedastic,
                      X_dependent_lambda=X_dependent_lambda,
                      lambda_start=lambda_start, c=c, gamma=gamma,
                      numSim=numSim, numIter=numIter, tol=tol,
                      threshold=threshold, par=par, corecap=corecap,
                      fix_seed=fix_seed)

        yr = reg1.est['residuals']

        reg2 = rlasso(x, d, post=post, colnames=colnames_x, intercept=intercept,
                      model=model, homoskedastic=homoskedastic,
                      X_dependent_lambda=X_dependent_lambda,
                      lambda_start=lambda_start, c=c, gamma=gamma,
                      numSim=numSim, numIter=numIter, tol=tol,
                      threshold=threshold, par=par, corecap=corecap,
                      fix_seed=fix_seed)

        dr = reg2.est['residuals']

        reg3 = lm(fit_intercept=True).fit(dr, yr)

        alpha = reg3.coef_[0,0]

        resid = yr - cvec(reg3.predict(dr))

        # This is a difference to the original code. The original code uses
        # var <- vcov(reg3)[2, 2], which is the homoskedastic covariance
        # estimator for OLS. I wrote get_cov() to calculate that, because the
        # linear regression implementation in sklearn does not include standard
        # error calculations. (I could have switched to statsmodels instead, but
        # sklearn seems more likely to be maintained in the future.) I then
        # added the option to get_cov() to calculate heteroskedastic standard
        # errors. I believe that if the penalty term is adjusted for
        # heteroskedasticity, heteroskedastic standard errors should also be
        # used here, to be internally consistent.
        var = np.array([get_cov(dr, resid, homoskedastic=homoskedastic)[1,1]])

        se = np.sqrt(var)

        tval = alpha / np.sqrt(var)

        pval = 2 * norm.cdf(-np.abs(tval))

        res = {'epsilon': resid, 'v': dr}

        I1 = reg1.est['index']

        I2 = reg2.est['index']

        I = cvec(I1.astype(bool) | I2.astype(bool))

        #names(I) <- union(names(I1),names(I2))

        results = {
            'alpha': alpha,
            'se': se,
            't': tval,
            'pval': pval,
            'coefficients': alpha,
            'coefficient': alpha,
            'coefficients_reg': reg1.est['coefficients'],
            'selection_index': I,
            'residuals': res,
            #call = match.call(),
            'samplesize': n
        }

    return results

################################################################################
### 3: Define classes
################################################################################


class rlassoEffects():
    # Initialize index to None to get index=c(1:ncol(x))
    def __init__(self, x, y, index=None, method='partialling out', I3=None,
                 post=True, colnames=None, intercept=True, model=True,
                 homoskedastic=False, X_dependent_lambda=False,
                 lambda_start=None, c=1.1, gamma=None, numSim=5000, numIter=15,
                 tol=10**(-5), threshold=-np.inf, par_outer=True,
                 par_inner=False, par_any=True, corecap=np.inf, fix_seed=True,
                 verbose=False):
        # Initialize internal variables
        if isinstance(x, pd.DataFrame) and colnames is None:
            colnames = x.columns

        self.x = np.array(x).astype(np.float32)
        self.y = cvec(y).astype(np.float32)

        if index is None:
            self.index = cvec(np.arange(self.x.shape[1]))
        else:
            self.index = cvec(index)

        self.method = method

        self.I3 = I3

        self.post = post

        self.colnames = colnames

        if self.index.dtype == bool:
            self.k = self.p1 = self.index.sum()
        else:
            self.k = self.p1 = len(self.index)

        self.n = x.shape[1]

        self.intercept = intercept
        self.model = model

        self.homoskedastic = homoskedastic
        self.X_dependent_lambda = X_dependent_lambda
        self.lambda_start = lambda_start

        self.c = c
        self.gamma = gamma

        self.numSim = numSim
        self.numIter = numIter
        self.tol = tol
        self.threshold = threshold

        self.par_outer = par_outer
        self.par_inner = par_inner
        self.par_any = par_any
        self.corecap = corecap
        self.fix_seed = fix_seed

        if not self.par_any:
            self.par_outer = self.par_inner = False
        elif self.par_outer and self.par_inner:
            self.par_outer = False

        self.verbose = verbose

        # Initialize internal variables used in other functions
        self.B = None
        self.parm = None
        self.level = None
        self.joint = None

        # preprocessing index numerical vector
        if np.issubdtype(self.index.dtype, np.number):
            self.index = self.index.astype(int)

            if not (np.all(self.index[:,0] < self.x.shape[1])
                    and (len(self.index) <= self.x.shape[1])):
                raise ValueError('Numeric index includes elements which are '
                                 + 'outside of the column range of x, or the '
                                 + 'indexing vector is too long')

        elif self.index.dtype == bool:
            if not (len(self.index) <= self.x.shape[1]):
                raise ValueError('Boolean index vector is too long')

            self.index = cvec([i for i, b in enumerate(self.index[:,0]) if b])

        elif np.issubdtype(self.index.dtype, np.str_):
            if not np.all([s in self.x.columns for s in self.index[:,0]]):
                raise ValueError('String index specifies column names which '
                                 + 'are not in the column names of x')

            self.index = (
                cvec([i for i, s in enumerate(self.index[:,0])
                      if s in self.x.columns])
            )

        else:
            raise ValueError('Argument index has an invalid type')

        if (self.method == 'double selection') and (self.I3 is not None):
            I3ind = cvec([i for i, b in enumerate(self.I3) if b])

            if I3ind != []:
                if len([x for x in I3ind[:,0] if x in self.index[:,0]]) > 0:
                    raise ValueError('I3 and index must not overlap!')

        if self.colnames is None:
            self.colnames = ['V' + str(i+1) for i in range(self.x.shape[1])]

        # Check whether to use parallel processing
        if self.par_outer:
            # If so, get the number of cores to use
            cores = int(np.amin([mp.cpu_count(), self.corecap]))
        else:
            # Otherwise, use only one core (i.e. run sequentially)
            cores = 1

        if (self.I3 is not None):
            res = jbl.Parallel(n_jobs=cores)(
                jbl.delayed(rlassoEffect_wrapper)(
                    i, x=np.delete(self.x, i, axis=1), y=self.y, d=self.x[:, i],
                    method=self.method, I3=np.delete(self.I3, i, axis=0),
                    post=self.post, colnames_d=self.colnames[i],
                    colnames_x=[c for j, c in enumerate(self.colnames) if j!=i],
                    intercept=self.intercept, model=self.model,
                    homoskedastic=self.homoskedastic,
                    X_dependent_lambda=self.X_dependent_lambda,
                    lambda_start=self.lambda_start, c=self.c, gamma=self.gamma,
                    numSim=self.numSim, numIter=self.numIter, tol=self.tol,
                    threshold=self.threshold, par=self.par_inner,
                    corecap=self.corecap, fix_seed=self.fix_seed,
                    verbose=self.verbose
                )
                for i in self.index[:,0]
            )
        else:
            res = jbl.Parallel(n_jobs=cores)(
                jbl.delayed(rlassoEffect_wrapper)(
                    i, x=np.delete(self.x, i, axis=1), y=self.y, d=self.x[:, i],
                    method=self.method, I3=self.I3,
                    post=self.post, colnames_d=self.colnames[i],
                    colnames_x=[c for j, c in enumerate(self.colnames) if j!=i],
                    intercept=self.intercept, model=self.model,
                    homoskedastic=self.homoskedastic,
                    X_dependent_lambda=self.X_dependent_lambda,
                    lambda_start=self.lambda_start, c=self.c, gamma=self.gamma,
                    numSim=self.numSim, numIter=self.numIter, tol=self.tol,
                    threshold=self.threshold, par=self.par_inner,
                    corecap=self.corecap, fix_seed=self.fix_seed,
                    verbose=self.verbose
                )
                for i in self.index[:,0]
            )

        # Convert collection of parallel results into usable results sorted by
        # their index
        coefficients = np.array([r['coefficients'] for r in res])
        coefficients = cvec(coefficients[coefficients[:,0].argsort(), 1])

        se = np.array([r['se'] for r in res])
        se = cvec(se[se[:,0].argsort(), 1])

        t = np.array([r['t'] for r in res])
        t = cvec(t[t[:,0].argsort(), 1])

        pval = np.array([r['pval'] for r in res])
        pval = cvec(pval[pval[:,0].argsort(), 1])

        lasso_regs = {}
        [lasso_regs.update(r['lasso_regs']) for r in res]

        reside = (
            np.array([np.concatenate([cvec(r['reside'][0]),
                                      r['reside'][1]],
                                     axis=0)[:,0]
                      for r in res])
        )
        reside = reside[reside[:,0].argsort(), 1:].T

        residv = (
            np.array([np.concatenate([cvec(r['residv'][0]),
                                      r['residv'][1]],
                                     axis=0)[:,0]
                      for r in res])
        )
        residv = residv[residv[:,0].argsort(), 1:].T

        coef_mat = {}
        [coef_mat.update(r['coef_mat']) for r in res]

        # Replaced this with the following two steps, to ensure this always
        # results in a two dimensional array
        #selection_matrix = (
        #    np.array([np.concatenate([cvec(r['selection_matrix'][0]),
        #                              r['selection_matrix'][1]],
        #                             axis=0)[:,0]
        #              for r in res])
        #)
        selection_matrix = [
            np.concatenate([cvec(r['selection_matrix'][0]),
                            r['selection_matrix'][1]],
                           axis=0).T
            for r in res
        ]
        selection_matrix = (
            np.concatenate(selection_matrix, axis=0)
        )
        selection_matrix = (
            selection_matrix[selection_matrix[:,0].argsort(), 1:]
        )

        # Added this, to be able to add names to results objects
        idx = [self.colnames[i] for i in self.index[:,0]]

        residuals = {
            'e': pd.DataFrame(reside, columns=idx),
            'v': pd.DataFrame(residv, columns=idx)
        }

        self.res = {
            'coefficients': pd.DataFrame(coefficients, index=idx),
            'se': pd.DataFrame(se, index=idx),
            't': pd.DataFrame(t, index=idx),
            'pval': pd.DataFrame(pval, index=idx),
            'lasso_regs': lasso_regs,
            'index': pd.DataFrame(self.index, index=idx),
            #call = match.call(),
            'samplesize': self.n,
            'residuals': residuals,
            'coef_mat': coef_mat,
            'selection_matrix': pd.DataFrame(selection_matrix, index=idx,
                                             columns=list(self.colnames))
        }

    def confint(self, parm=None, B=500, level=.95, joint=False,
                par=None, corecap=None, fix_seed=None, verbose=None):
        self.B = B

        if par is None:
            par = self.par_any
        
        if corecap is None:
            corecap = self.corecap
        
        if fix_seed is None:
            fix_seed = self.fix_seed
        
        if verbose is None:
            verbose = self.verbose
        
        n = self.res['samplesize']
        
        k = p1 = len(self.res['coefficients'])
        
        cf = self.res['coefficients']
        
        pnames = cf.index.values
        
        self.parm = parm
        self.level = level
        self.joint = joint
        
        if self.parm is None:
            self.parm = pnames
        elif np.issubdtype(self.parm, np.number):
            self.parm = pnames[parm]
        
        if not self.joint:
            a = (1 - self.level) / 2
            
            a = cvec([a, 1 - a])
            
            fac = norm.ppf(a)
            
            pct = [str(np.round(x * 100, 3)) + ' %' for x in a[:,0]]
            
            ses = self.res['se'].loc[self.parm, :]
            
            self.ci = cf.loc[self.parm, :] @ np.ones(shape=(1, 2)) + ses @ fac.T
            
            self.ci.columns = pct
            
            return self.ci
        else:
            if self.verbose:
                print('\nCaution: Joint confidence intervals for hdmpy are',
                      'currently different from those of the original R',
                      'package hdm. This is a known bug.')
            e = self.res['residuals']['e'].values
            v = self.res['residuals']['v'].values
            
            ev = e * v
            
            Ev2 = np.mean(v**2, axis=0)
            
            Omegahat = np.zeros(shape=(self.k, self.k)) * np.nan
            
            for j in np.arange(self.k):
                for l in np.arange(start=j, stop=self.k):
                    Omegahat[j,l] = Omegahat[l,j] = (
                        1/(Ev2[j] * Ev2[l]) * np.mean(ev[:,j] * ev[:,l])
                    )
            
            var = np.diag(Omegahat)
            
            # Check whether to use parallel processing
            if par:
                # If so, get the number of cores to use
                cores = int(np.amin([mp.cpu_count(), self.corecap]))
            else:
                # Otherwise, use only one core (i.e. run sequentially)
                cores = 1
            
            sim = jbl.Parallel(n_jobs=cores)(
                jbl.delayed(simul_ci)(
                    Omega=Omegahat/self.n, var=var, seed=i*20,
                    fix_seed=fix_seed, verbose=verbose
                )
                for i in np.arange(self.B)
            )
            
            sim = cvec(sim)
            
            a = 1 - self.level
            
            ab = cvec([a/2, 1 - a/2])
            
            pct = [str(np.round(x * 100, 3)) + ' %' for x in ab[:,0]]
            
            var = pd.DataFrame(var, index=self.parm)
            
            hatc = np.quantile(sim, q=1-a)
            
            ci1 = cf.loc[self.parm, :] - hatc * np.sqrt(var.loc[self.parm, :])
            
            ci2 = cf.loc[self.parm, :] + hatc * np.sqrt(var.loc[self.parm, :])
            
            self.ci = pd.concat([ci1.iloc[:,0], ci2.iloc[:,0]], axis=1)
            
            self.ci.columns = pct
            
            return self.ci


