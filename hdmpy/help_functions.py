################################################################################
### Python port of help_functions.R
### https://github.com/cran/hdm/blob/master/R/help_functions.R
################################################################################

################################################################################
### 1: Load modules
################################################################################

# Standard Python modules
import numpy as np
from sklearn.linear_model import LinearRegression as lm

################################################################################
### 2: Define functions
################################################################################

################################################################################
### 2.1: Functions which are not in the original R package
###      These are generally helper functions to allow an implementation which
###      reads as closely to the original R code as possible, and to ease a
###      Python implementation
################################################################################


# Define a function which turn a list or vector-like object into a proper two
# dimensional column vector
def cvec(a):
    """ Turn a list or vector-like object into a proper column vector

    Input
    a: List or vector-like object, has to be a potential input for np.array()

    Output
    vec: two dimensional NumPy array, with the first dimension weakly greater
         than the second (resulting in a column vector for a vector-like input)
    """
    # Conver input into a two dimensional NumPy array
    vec = np.array(a, ndmin=2)

    # Check whether the second dimension is strictly greater than the first
    # (remembering Python's zero indexing)
    if vec.shape[0] < vec.shape[1]:
        # If so, transpose the input vector
        vec = vec.T

    # Return the column vector
    return vec


# Define a function to mimic R's cor() function, which can take two matrices and
# return the correlation coefficients between the columns of the first and the
# columns of the second matrix
def cor(y, X):
    """ Return correlation coefficients between columns of matrices

    Inputs
    y: n by 1 NumPy array
    X: n by k NumPy array

    Outputs
    corr: list of length k, where the k-th element is the correlation
          coefficient between y and the k-th column of X
    """
    # Concatenate y and X into a single NumPy array
    yX = np.concatenate([y, X], axis=1)

    # Get the correlation coefficients between all columns of that array
    corr = np.corrcoef(yX, rowvar=False)

    # Get the first row, starting at the first off-diagonal element (these are
    # the correlation coefficients between y and each column of X
    corr = corr[0,1:]

    # Return the result
    return corr

################################################################################
### 2.2: Functions which are in the original R package
################################################################################


# Define a function which returns initial parameter guesses
def init_values(X, y, number=5, intercept=True):
    """ Return an initial parameter guess for a LASSO model

    Inputs
    y: n by 1 NumPy array, outcome variable
    X: n by k NumPy array, RHS variables

    Outputs
    residuals: n ny 1 NumPy array, residuals for initial parameter guess
    coefficients: k by 1 NumPy array, initial coefficient values
    """
    # Make sure y is a proper column vector
    y = cvec(y)

    # Get the absolute value of correlations between y and X
    corr = np.abs(cor(y, X))

    # Get the number of columns of X
    kx = X.shape[1]

    # Make an index selecting the five columns of X which are most correlated
    # with y (since .argsort() always sorts in increasing order, selecting from
    # the back gets the most highly correlated columns)
    index = corr.argsort()[-np.amin([number, kx]):]

    # Set up an array of coefficient guesses
    coefficients = np.zeros(shape=(kx, 1))

    # Regress y on the five most correlated columns of X, including an intercept
    # if desired
    reg = lm(fit_intercept=intercept).fit(X[:, index], y)

    # Replace the guesses for the estimated coefficients (note that .coef_ does
    # not return the estimated intercept, if one was included in the model)
    coefficients[index, :] = reg.coef_.T

    # Replace any NANs as zeros
    coefficients[np.isnan(coefficients)] = 0

    # Get the regression residuals
    residuals = y - reg.predict(X[:, index])

    # Return the residuals and coefficients
    return {'residuals': residuals, 'coefficients': coefficients}


