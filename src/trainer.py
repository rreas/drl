class Trainer:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def train(self, window, lookback, slide, maxiter=2000):
        start = 0
        trX, trY, tsX, tsY = self.dataset.build(start, window, lookback, slide)
        weights = self.model.weights(trX, seed=1)

        for i in range(maxiter):
            # Update weights with SGD.
            cost = self.model.cost(weights, trX, trY)
            gradient = self.model.grad(weights, trX, trY)
            weights = weights - gradient

            print "Iter ", i, ": ", self.model.mean_return(weights, tsX, tsY)
