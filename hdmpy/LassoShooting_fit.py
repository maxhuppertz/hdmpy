################################################################################
### Python port of LassoShooting.fit.R
### https://github.com/cran/hdm/blob/master/R/LassoShooting.fit.R
################################################################################

################################################################################
### 1: Load modules
################################################################################

# Standard Python modules
import numpy as np

# Other parts of hdmpy
from hdmpy.help_functions import cvec, init_values

################################################################################
### 2: Define function
################################################################################

# Define shooting LASSO with variable dependent penalty terms
def LassoShooting_fit(x, y, lmbda, maxIter=1000, optTol=10**(-5),
                      zeroThreshold=10**(-6), XX=None, Xy=None,
                      beta_start=None):
    """ Shooting LASSO algorithm with variable dependent penalty weights

    Inputs
    x: n by p NumPy array, RHS variables
    y: n by 1 NumPy array, outcome variable
    lmbda: p by 1 NumPy array, variable dependent penalty terms. The j-th
           element is the penalty term for the j-th RHS variable.
    maxIter: integer, maximum number of shooting LASSO updated
    optTol: scalar, algorithm terminated once the sum of absolute differences
            between the updated and current weights is below optTol
    zeroThreshold: scalar, if any final weights are below zeroThreshold, they
                   will be set to zero instead
    XX: k by k NumPy array, pre-calculated version of x'x
    Xy: k by 1 NumPy array, pre-calculated version of x'y
    beta_start: k by 1 NumPy array, initial weights

    Outputs
    w: k by 1 NumPy array, final weights
    wp: k by m + 1 NumPy array, where m is the number of iterations the
        algorithm took. History of weight updates, starting with the initial
        weights.
    m: integer, number of iterations the algorithm took
    """
    # Make sure that y and lmbda are proper column vectors
    y = cvec(y)
    lmbda = cvec(lmbda)

    # Get number of observations n and number of variables p
    n, p = x.shape

    # Check whether XX and Xy were provided, calculate them if not
    if XX is None:
        XX = x.T @ x
    if Xy is None:
        Xy = x.T @ y

    # Check whether an initial value for the intercept was provided
    if beta_start is None:
        # If not, use init_values from help_functions, which will return
        # regression estimates for the five variables in x which are most
        # correlated with y, and initialize all other coefficients as zero
        beta = init_values(x, y, intercept=False)['coefficients']
    else:
        # Otherwise, use the provided initial weights
        beta = beta_start

    # Set up a history of weights over time, starting with the initial ones
    wp = beta

    # Keep track of the number of iterations
    m = 1

    # Create versions of XX and Xy which are just those matrices times two
    XX2 = XX * 2
    Xy2 = Xy * 2

    # Go through all iterations
    while m < maxIter:
        # Save the last set of weights (the .copy() is important, otherwise
        # beta_old will be updated every time beta is changed during the
        # following loop)
        beta_old = beta.copy()

        # Go through all parameters
        for j in np.arange(p):
            # Calculate the shoot
            S0 = XX2[j,:] @ beta - XX2[j,j] * beta[j,0] - Xy2[j,0]

            # Update the weights
            if np.isnan(S0).sum() >= 1:
                beta[j] = 0
            elif S0 > lmbda[j]:
                beta[j] = (lmbda[j] - S0) / XX2[j,j]
            elif S0 < -lmbda[j]:
                beta[j] = (-lmbda[j] - S0) / XX2[j,j]
            elif np.abs(S0) <= lmbda[j]:
                beta[j] = 0

        # Add the updated weights to the history of weights
        wp = np.concatenate([wp, beta], axis=1)

        # Check whether the weights are within tolerance
        if np.abs(beta - beta_old).sum() < optTol:
            # If so, break the while loop
            break

        # Increase the iteration counter
        m = m + 1

    # Set the final weights to the last updated weights
    w = beta

    # Set weights which are within zeroThreshold to zero
    w[np.abs(w) < zeroThreshold] = 0

    # Return the weights, history of weights, and iteration counter
    return {'coefficients': w, 'coef.list': wp, 'num.it': m}


