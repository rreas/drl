from numpy import sqrt, power, zeros, array, std, mean, append
from scipy.stats import zscore
from scipy.optimize import minimize

from utils import sharpe, wealth

class Trainer:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def scale(self, trX, tsX):
        for i in range(trX.shape[1]):
            vec = trX[:,i]
            m = mean(vec)
            s = std(vec)

            if s == 0:
                continue # Assume bias column so no need to scale.

            trX[:,i] = (trX[:,i] - m) / s
            tsX[:,i] = (tsX[:,i] - m) / s

    def train(self, window=100, lookback=5, slide=50, maxiter=100):
        returns = array([])
        for trX, trY, tsX, tsY in self.dataset.gen(window=window,
                lookback=lookback, slide=slide):
            returns = append(returns, self.run(trX, trY, tsX, tsY, maxiter))

        return returns

class AdaGradTrainer(Trainer):

    def run(self, trX, trY, tsX, tsY, maxiter):
        self.scale(trX, tsX)

        weights = self.model.weights(trX, seed=1)
        gradients = zeros(weights.shape)

        trR = zeros(maxiter+1)
        tsR = zeros(maxiter+1)

        for i in range(maxiter):
            # Update weights with SGD.
            cost = self.model.cost(weights, trX, trY)
            gradient = self.model.grad(weights, trX, trY)
            gradients += power(gradient, 2)
            rate = 1. / sqrt(gradients)
            weights = weights - rate*gradient

        returns = self.model.returns(weights, tsX, tsY)

        s = sharpe(returns)
        w = wealth(returns)
        print "Test: Sharpe %.10f, Wealth %f" % (s[-1], w[-1])

        return returns

