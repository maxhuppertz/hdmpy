################################################################################
### Python port of rlasso.R
### https://github.com/cran/hdm/blob/master/R/rlasso.R
################################################################################

################################################################################
### 1: Load modules
################################################################################

# Standard Python modules
import joblib as jbl
import multiprocess as mp
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression as lm

# Other parts of hdmpy
from hdmpy.help_functions import cvec, init_values
from hdmpy.LassoShooting_fit import LassoShooting_fit

################################################################################
### 2: Define functions
################################################################################

################################################################################
### 2.1: Functions which are not in the original R package
###      These are generally helper functions to allow an implementation which
###      reads as closely to the original R code as possible, and to ease a
###      Python implementation, including parallelizing the code
################################################################################


# Define a function to simulate distributions needed for calculating X-dependent
# penalty terms
def simul_pen(n, p, W, seed=0, fix_seed=True):
    # Check whether the seed needs to be fixed
    if fix_seed:
        # Simulate with provided seed
        g = norm.rvs(size=(n,1), random_state=seed) @ np.ones(shape=(1,p))
    else:
        # Simulate using whatever state the RNG is currently in
        g = norm.rvs(size=(n,1)) @ np.ones(shape=(1,p))

    # Calculate element of the distribution for the current draw of g
    s = n * np.amax(2 * np.abs(np.mean(W * g, axis=0)))

    # Return the result
    return s

################################################################################
### 2.2: Functions which are in the original R package
################################################################################


def lambdaCalculation(homoskedastic=False, X_dependent_lambda=False,
                      lambda_start=None, c=1.1, gamma=0.1, numSim=5000, y=None,
                      x=None, par=True, corecap=np.inf, fix_seed=True):
    # Get number of observations n and number of variables p
    n, p = x.shape

    # Get number of simulations to use (if simulations are necessary)
    R = numSim

    # Go through all possible combinations of homoskedasticy/heteroskedasticity
    # and X-dependent or independent error terms. The first two cases are
    # special cases: Handling the case there homoskedastic was set to None, and
    # where lambda_start was provided.
    #
    # 1) If homoskedastic was set to None (special case)
    if homoskedastic is None:
        # Initialize lambda
        lmbda0 = lambda_start

        Ups0 = (1 / np.sqrt(n)) * np.sqrt((y**2).T @ (x**2)).T

        # Calculate the final vector of penalty terms
        lmbda = lmbda0 * Ups0

    # 2) If lambda_start was provided (special case)
    elif lambda_start is not None:
        # Check whether a homogeneous penalty term was provided (a scalar)
        if np.amax(cvec(lambda_start).shape) == 1:
            # If so, repeat that p times as the penalty term
            lmbda = np.ones(shape=(p,1)) * lambda_start
        else:
            # Otherwise, use the provided vector of penalty terms as is
            lmbda = lambda_start

    # 3) Homoskedastic and X-independent
    elif (homoskedastic == True) and (X_dependent_lambda == False):
        # Initilaize lambda
        lmbda0 = 2 * c * np.sqrt(n) * norm.ppf(1 - gamma/(2*p))

        # Use ddof=1 to be consistent with R's var() function
        Ups0 = np.sqrt(np.var(y, axis=0, ddof=1))

        # Calculate the final vector of penalty terms
        lmbda = np.zeros(shape=(p,1)) + lmbda0 * Ups0

    # 4) Homoskedastic and X-dependent
    elif (homoskedastic == True) and (X_dependent_lambda == True):
        psi = cvec((x**2).mean(axis=0))

        tXtpsi = (x.T / np.sqrt(psi)).T

        # Check whether to use parallel processing
        if par == True:
            # If so, get the number of cores to use
            cores = int(np.amin([mp.cpu_count(), corecap]))
        else:
            # Otherwise, use only one core (i.e. run sequentially)
            cores = 1

        # Get simulated distribution
        sim = jbl.Parallel(n_jobs=cores)(
            jbl.delayed(simul_pen)(
                n, p, tXtpsi, seed=l*20, fix_seed=fix_seed
            ) for l in np.arange(R)
        )

        # Convert it to a proper column vector
        sim = cvec(sim)

        # Initialize lambda based on the simulated quantiles
        lmbda0 = c * np.quantile(sim, q=1-gamma, axis=0)

        Ups0 = np.sqrt(np.var(y, axis=0, ddof=1))

        # Calculate the final vector of penalty terms
        lmbda = np.zeros(shape=(p,1)) + lmbda0 * Ups0

    # 5) Heteroskedastic and X-independent
    elif (homoskedastic == False) and (X_dependent_lambda == False):
        # The original includes the comment, "1=num endogenous variables"
        lmbda0 = 2 * c * np.sqrt(n) * norm.ppf(1 - gamma/(2*p*1))

        Ups0 = (1 / np.sqrt(n)) * np.sqrt((y**2).T @ (x**2)).T

        lmbda = lmbda0 * Ups0

    # 6) Heteroskedastic and X-dependent
    elif (homoskedastic == False) and (X_dependent_lambda == True):
        eh = y

        ehat = eh @ np.ones(shape=(1,p))

        xehat = x * ehat

        psi = cvec((xehat**2).mean(axis=0)).T

        tXehattpsi = (xehat / ( np.ones(shape=(n,1)) @ np.sqrt(psi) ))

        # Check whether to use parallel processing
        if par == True:
            # If so, get the number of cores to use
            cores = int(np.amin([mp.cpu_count(), corecap]))
        else:
            # Otherwise, use only one core (i.e. run sequentially)
            cores = 1

        # Get simulated distribution
        sim = jbl.Parallel(n_jobs=cores)(
            jbl.delayed(simul_pen)(
                n, p, tXehattpsi, seed=l*20, fix_seed=fix_seed
            ) for l in np.arange(R)
        )

        # Convert it to a proper column vector
        sim = cvec(sim)

        # Initialize lambda based on the simulated quantiles
        lmbda0 = c * np.quantile(sim, q=1-gamma, axis=0)

        Ups0 = (1 / np.sqrt(n)) * np.sqrt((y**2).T @ (x**2)).T

        # Calculate the final vector of penalty terms
        lmbda = lmbda0 * Ups0

    # Return results
    return {'lambda0': lmbda0, 'lambda': lmbda, 'Ups0': Ups0}

################################################################################
### 3: Define classes
################################################################################

class rlasso:
    # Initialize gamma to None to get gamma=.1/log(n)
    def __init__(self, x, y, colnames=None, post=True, intercept=True,
                 model=True, homoskedastic=False, X_dependent_lambda=False,
                 lambda_start=None, c=1.1, gamma=None, numSim=5000, numIter=15,
                 tol=10**(-5), threshold=-np.inf, par=True, corecap=np.inf,
                 fix_seed=True):
        # Initialize internal variables
        if isinstance(x, pd.DataFrame) and colnames is None:
            colnames = x.columns

        self.x = np.array(x).astype(np.float32)
        self.y = cvec(y).astype(np.float32)

        self.n, self.p = self.x.shape

        if colnames is None:
            self.colnames = ['V' + str(i+1) for i in np.arange(self.p)]
        else:
            self.colnames = colnames

        # Unused line in the original code
        # ind_names = np.arange(self.p) + 1

        self.post = post
        self.intercept = intercept
        self.model = model
        self.homoskedastic = homoskedastic
        self.X_dependent_lambda = X_dependent_lambda
        self.lambda_start = lambda_start
        self.c = c

        if gamma is None:
            self.gamma = .1 / np.log(self.n)
        else:
            self.gamma = gamma

        self.numSim = numSim
        self.numIter = numIter
        self.tol = tol
        self.threshold = threshold

        self.par = par
        self.corecap = corecap
        self.fix_seed = fix_seed

        if (self.post == False) and (self.c is None):
            self.c = .5

        if (
                (self.post == False) and (self.homoskedastic == False)
                and (self.X_dependent_lambda == False)
                and (self.lambda_start == None) and (self.c == 1.1)
                and (self.gamma == .1 / np.log(self.n))
        ):
            self.c = .5

        # For now, instantiate estimate as None
        self.est = None

        # Calculate robust LASSO coefficients
        if self.intercept == True:
            meanx = cvec(self.x.mean(axis=0))

            self.x = self.x - np.ones(shape=(self.n,1)) @ meanx.T

            mu = self.y.mean()

            self.y = self.y - mu
        else:
            meanx = np.zeros(shape=(self.p,1))

            mu = 0

        normx = np.sqrt(np.var(self.x, axis=1, ddof=1))

        Psi = cvec(np.mean(self.x**2, axis=0))

        ind = np.zeros(shape=(self.p,1)).astype(bool)

        XX = self.x.T @ self.x

        Xy = self.x.T @ self.y

        startingval = init_values(self.x, self.y)['residuals']

        pen = lambdaCalculation(homoskedastic=self.homoskedastic,
                                X_dependent_lambda=self.X_dependent_lambda,
                                lambda_start=self.lambda_start, c=self.c,
                                gamma=self.gamma, numSim=self.numSim,
                                y=startingval, x=self.x, par=self.par,
                                corecap=self.corecap, fix_seed=self.fix_seed)

        lmbda = pen['lambda']
        Ups0 = Ups1 = pen['Ups0']
        lmbda0 = pen['lambda0']

        mm = 1
        s0 = np.sqrt(np.var(y, axis=0, ddof=1))

        while mm <= self.numIter:
            if (mm == 1) and self.post:
                coefTemp = (
                    LassoShooting_fit(self.x, self.y, lmbda/2, XX=XX,
                                      Xy=Xy)['coefficients']
                )
            else:
                coefTemp = (
                    LassoShooting_fit(self.x, self.y, lmbda, XX=XX,
                                      Xy=Xy)['coefficients']
                )

            coefTemp[np.isnan(coefTemp)] = 0

            ind1 = (np.abs(coefTemp) > 0)

            x1 = self.x[:, ind1[:,0]]

            if x1.shape[1] == 0:
                if self.intercept:
                    intercept_value = np.mean(self.y + mu)

                    coef = np.zeros(shape=(self.p+1,1))

                    coef = (
                        pd.DataFrame(coef,
                                     index=['(Intercept)']+list(self.colnames))
                    )
                else:
                    intercept_value = np.mean(self.y)

                    coef = np.zeros(shape=(self.p,1))

                    coef = pd.DataFrame(coef, index=self.colnames)

                self.est = {
                    'coefficients': coef,
                    'beta': np.zeros(shape=(self.p,1)),
                    'intercept': intercept_value,
                    'index': pd.DataFrame(np.zeros(shape=(self.p,1)).astype(
                        bool),
                                          index=self.colnames),
                    'lambda': lmbda,
                    'lambda0': lmbda0,
                    'loadings': Ups0,
                    'residuals': self.y - np.mean(self.y),
                    'sigma': np.var(self.y, axis=0, ddof=1),
                    'iter': mm,
                    #'call': Not a Python option
                    'options': {'post': self.post, 'intercept': self.intercept,
                                'ind.scale': ind, 'mu': mu, 'meanx': meanx}
                }

                if self.model:
                    self.est['model'] = self.x
                else:
                    self.est['model'] = None

                self.est['tss'] = self.est['rss'] = (
                    ((self.y - np.mean(self.y))**2).sum()
                )

                self.est['dev']: self.y - np.mean(self.y)
                # In R, return() breaks while loops
                return

            # Refinement variance estimation
            if self.post:
                reg = lm(fit_intercept=False).fit(x1, self.y)

                coefT = reg.coef_.T

                coefT[np.isnan(coefT)] = 0

                e1 = self.y - x1 @ coefT

                coefTemp[ind1[:,0]] = coefT
            else:
                e1 = self.y - x1@ coefTemp[ind1[:,0]]

            s1 = np.sqrt(np.var(e1, ddof=1))

            # Homoskedastic and X-independent
            if (
                    (self.homoskedastic == True)
                    and (self.X_dependent_lambda == False)
            ):
                Ups1 = s1 * Psi

                lmbda = pen['lambda0'] * Ups1

            # Homoskedastic and X-dependent
            elif (
                    (self.homoskedastic == True)
                    and (self.X_dependent_lambda == True)
            ):
                Ups1 = s1 * Psi

                lmbda = pen['lambda0'] * Ups1

            # Heteroskedastic and X-independent
            elif (
                    (self.homoskedastic == False)
                    and (self.X_dependent_lambda == False)
            ):
                Ups1 = (
                    (1/np.sqrt(self.n)) * np.sqrt((e1**2).T @ self.x**2).T
                )

                lmbda = pen['lambda0'] * Ups1

            # Heteroskedastic and X-dependent
            elif (
                    (self.homoskedastic == False)
                    and (self.X_dependent_lambda == True)
            ):
                lc = lambdaCalculation(homoskedastic=self.homoskedastic,
                                       X_dependent_lambda=
                                       self.X_dependent_lambda,
                                       lambda_start=self.lambda_start,
                                       c=self.c, gamma=self.gamma,
                                       numSim=self.numSim, y=e1, x=self.x,
                                       par=self.par, corecap=self.corecap,
                                       fix_seed=self.fix_seed)

                Ups1 = lc['Ups0']

                lmbda = lc['lambda']

            # If homoskedastic is set to None
            elif self.homoskedastic is None:
                Ups1 = (
                    (1/np.sqrt(self.n)) * np.sqrt((e1**2).T @ self.x**2).T
                )

                lmbda = pen['lambda0'] * Ups1

            mm = mm + 1

            if np.abs(s0 - s1) < self.tol:
                break

            s0 = s1

        if x1.shape[1] == 0:
            #coefTemp = None
            ind1 = np.zeros(shape=(self.p,1))

        coefTemp = cvec(coefTemp)

        coefTemp[np.abs(coefTemp) < self.threshold] = 0

        coefTemp = pd.DataFrame(coefTemp, index=self.colnames)

        ind1 = cvec(ind1)

        ind1 = pd.DataFrame(ind1, index=self.colnames)

        if self.intercept:
            if mu is None:
                mu = 0
            if meanx is None:
                meanx = np.zeros(shape=(coefTemp.shape[0],1))
            if ind.sum() == 0:
                intercept_value = mu - (meanx * coefTemp).sum()
            else:
                intercept_value = mu - (meanx * coefTemp).sum()
        else:
            intercept_value = np.nan

        if self.intercept:
            beta = (
                np.concatenate([cvec(intercept_value), coefTemp.values], axis=0)
            )

            beta = pd.DataFrame(beta, index=['(Intercept)']+list(self.colnames))
        else:
            beta = coefTemp

        s1 = np.sqrt(np.var(e1, ddof=1))

        self.est = {
            'coefficients': beta,
            'beta': pd.DataFrame(coefTemp, index=self.colnames),
            'intercept': intercept_value,
            'index': ind1,
            'lambda': pd.DataFrame(lmbda, index=self.colnames),
            'lambda0': lmbda0,
            'loadings': Ups1,
            'residuals': cvec(e1),
            'sigma': s1,
            'iter': mm,
            #'call': Not a Python option
            'options': {'post': self.post, 'intercept': self.intercept,
                        'ind.scale': ind, 'mu': mu, 'meanx': meanx},
            'model': model
        }

        if model:
            self.x = self.x + np.ones(shape=(self.n,1)) @ meanx.T

            self.est['model'] = self.x
        else:
            self.est['model'] = None

        self.est['tss'] = ((self.y - np.mean(self.y))**2).sum()
        self.est['rss'] = (self.est['residuals']**2).sum()
        self.est['dev'] = self.y - np.mean(self.y)


