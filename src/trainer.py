from numpy import sqrt, power, zeros, array, std, mean, append, vstack
from scipy.stats import zscore
from scipy.optimize import minimize
from numpy.random import RandomState

import matplotlib.pyplot as plt

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
        decisions = array([])

        for trX, trY, tsX, tsY in self.dataset.gen(window=window,
                lookback=lookback, slide=slide):

            r, d = self.run(trX, trY, tsX, tsY, maxiter)
            returns = append(returns, r)
            decisions = append(decisions, r)

        return returns, decisions

class AdaGradTrainer(Trainer):

    def run(self, trX, trY, tsX, tsY, maxiter, validating=False):
        self.scale(trX, tsX)

        weights = self.model.weights(trX, seed=1)
        gradients = zeros(weights.shape)

        trR = zeros(maxiter+1)
        tsR = zeros(maxiter+1)

        costs = []

        for i in range(maxiter):
            #rs = RandomState(i)
            #r = rs.randint(0, trX.shape[0]/2)
            #tempX = trX[r:, :]
            #tempY = trY[r:]

            gradient = self.model.grad(weights, trX, trY)
            gradients += power(gradient, 2)
            rate = 1. / sqrt(gradients)
            weights = weights - rate*gradient

            if validating:
                costs.append(self.model.cost(weights, tsX, tsY))

        # Calculate returns and decisions on test set.
        returns, decisions = self.model.returns(weights, tsX, tsY)

        s = sharpe(returns)
        w = wealth(returns)

        return returns, decisions

class ValidatingTrainer(AdaGradTrainer):

    def __init__(self, dataset, models):
        Trainer.__init__(self, dataset, None)
        self.models = models

    def train(self, window=100, lookback=5, slide=50, maxiter=100):
        returns = array([])
        decisions = array([])

        self.dataset.reset(window=window, lookback=lookback, slide=slide)
        trX, trY, vsX, vsY = self.dataset.gen()

        while(self.dataset.can_gen()):

            score = float('-inf')
            best = None

            for model in self.models:
                self.model = model
                r, _ = self.run(trX, trY, vsX, vsY, maxiter, validating=True)
                w = wealth(r)[-1]

                if w > score:
                    score = w
                    best = model

            next_trX, next_trY, tsX, tsY = self.dataset.gen()
            self.model = best

            r, d = self.run(next_trX, next_trY, tsX, tsY, maxiter)
            returns = append(returns, r)
            decisions = append(decisions, d)

            print "Wealth: %f\tModel: %s" % (wealth(r)[-1], self.model)

            # Done with loop carry over to next iteration.
            trX = next_trX
            trY = next_trY
            vsX = tsX
            vsY = tsY

        return returns, decisions

