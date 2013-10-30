from numpy import tanh, append, dot, reshape, absolute
from numpy.random import randn
from scipy.optimize import minimize

from dataset import Dataset
from utils import sharpe

class Network:

    def __init__(self, raw, lookback=10, size=100, hidden=5):
        self.dataset = Dataset(raw)
        self.lookback = lookback
        self.size = size
        self.hidden = hidden
        self.delta = 0.01

    def initial_weights(self):
        self.h = randn(self.hidden)
        self.W = randn(self.lookback+2, self.hidden)
        return self.compress(self.h, self.W)

    def compress(self, h, W):
        return append(h, reshape(W, W.size), axis=0)

    def uncompress(self, params):
        h = params[0:self.hidden]
        W = reshape(params[self.hidden:], self.W.shape)
        return h, W

    def pred(self, params, x):
        h, W = self.uncompress(params)
        return tanh(dot(h, tanh(dot(W.T, x))))

    def train(self):
        returns = []

        for x, y in self.dataset.gen(size=self.size, lookback=self.lookback):
            # Train model on available data
            minimize(self.cost,
                     self.initial_weights(),
                     (x, y),
                     options={'disp': True})

            # Make a single prediction


    def cost(self, params, x, y):
        previous = 0.
        returns = []

        for i in range(0, len(x)):
            example = append(x[i], [previous, 1.])
            pred = self.pred(params, example)
            returns.append(pred*y[i] - self.delta*absolute(pred-previous))
            previous = pred

        return -1. * sharpe(returns)


# previous day
# bias
# cost function = negative sharpe
