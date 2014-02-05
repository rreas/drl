from numpy import sqrt, power, zeros, array, std, mean
from scipy.stats import zscore

class Trainer:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def train(self, window, lookback, slide, maxiter=50):
        start = 0
        trX, trY, tsX, tsY = self.dataset.build(start, window, lookback, slide)
        
        # Bias is first element.
        #for i in range(1, trX.shape[1]):
        #    m = mean(trX[:,i])
        #    s = std(trX[:,i])

        #    trX[:,i] = (trX[:,i] - m) / s
        #    tsX[:,i] = (tsX[:,i] - m) / s

        weights = self.model.weights(trX, seed=1)
        gradients = zeros(weights.shape)

        for i in range(maxiter):
            # Update weights with SGD.
            cost = self.model.cost(weights, trX, trY)
            gradient = self.model.grad(weights, trX, trY)
            gradients += power(gradient, 2)

            rate = 1. / sqrt(gradients)

            weights = weights - rate*gradient

            print "Iter ", i+1, ": ", self.model.mean_return(weights, tsX, tsY)

        print weights

        return weights

