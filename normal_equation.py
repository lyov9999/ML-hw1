import numpy as np
from numpy.linalg import inv

def linear_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression to the data of form Y = w * X
    **Note** that there is not separate term for the intercept.

    Example
    >>> linear_regression(np.array([[1, 1], [1, 6]]), np.array([1, 2]))
    array([0.8, 0.2])
    """
    X_transpose = X.T 
    w = inv(X_transpose.dot(X)).dot(X_transpose).dot(Y)
    return w 

    
