from numpy import mean, std, zeros, max, min, exp
from numpy.random import RandomState
from numpy.linalg import norm

def sharpe(returns):
    return mean(returns) / std(returns)

def cost(returns, params):
    return -1*sharpe(returns) + norm(params)**2

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
