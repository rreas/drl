from numpy import (mean, std, zeros, max, min, exp, tanh, log, sum, cumsum,
        append)

from numpy.random import RandomState
from numpy.linalg import norm

def sech2(data):
    return 1 - tanh(data)**2

def sign(value):
    return -1. if value < 0 else 1.

def sharpe(returns):
    s = zeros(len(returns))
    s[0] = returns[0]

    for i in range(1, len(returns)):
        mu = mean(returns[:i+1])
        sigma = std(returns[:i+1])

        if sigma == 0:
           continue 

        s[i] = mu / sigma

    return s

def wealth(returns):
    return exp(cumsum(log(1 + returns)))

def synthetic(n, phi=0.9, k=3, var=1, seed=1):
    """Auto-regressive test data generator."""
    prng = RandomState(seed)
    P = zeros((n,))
    X = zeros((n,))

    # Initial values for P, X.
    P[0] = prng.normal(0,var)
    X[0] = prng.normal(0, var)

    for i in xrange(1, n):
        P[i] = P[i-1] + X[i-1] + k*prng.normal(0, var)
        X[i] = phi*X[i-1] + prng.normal(0, var)

    R = max(P) - min(P)
    Z = exp(P/R)

    return Z
