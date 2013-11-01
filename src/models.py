from numpy import tanh, append, dot, reshape, absolute
from numpy.random import RandomState
from numpy.linalg import norm

from utils import sharpe

class Linear:

    def __init__(self, lookback=10, delta=0.01, lmb=1.):
        self.lookback = lookback
        self.delta = delta
        self.lmb = lmb

    def initial_weights(self):
        rs = RandomState(1)
        self.w = rs.randn(lookback+2)
        return self.w

    def cost(self, params, x, y):
        previous = 0.
        gradient = zeros()
        returns = []

        return None

    def pred(self, params, x):
        raise "Not implemented"

class Nonlinear:

    def __init__(self, hidden=10, lookback=10, delta=0.01, lmb=1.):
        self.hidden = hidden
        self.lookback = lookback
        self.delta = 0.01
        self.lmb = 1.
        self.shape_w = (self.lookback+2, self.hidden)

    def initial_weights(self, seed=1):
        rs = RandomState(1)
        self.h = rs.randn(self.hidden)
        self.W = rs.randn(self.lookback+2, self.hidden)
        return self.compress(self.h, self.W)

    def retval(self, y, pred, previous):
        return pred*y - self.delta*absolute(pred-previous)

    def pred(self, params, x):
        h, W = self.uncompress(params)
        return tanh(dot(h, tanh(dot(W.T, x))))

    def compress(self, h, W):
        return append(h, reshape(W, W.size), axis=0)

    def uncompress(self, params):
        h = params[0:self.hidden]
        W = reshape(params[self.hidden:], self.W.shape)
        return h, W

    def cost(self, params, x, y):
        previous = 0.
        returns = []

        for i in range(0, len(x)):
            example = append(x[i], [previous, 1.])
            pred = self.pred(params, example)
            returns.append(self.retval(y[i], pred, previous))
            previous = pred

        return -1.*sharpe(returns) + self.lmb*norm(params)
