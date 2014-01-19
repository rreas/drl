class Trainer:

    def __init__(self, dataset, model, window=100, lookback=10, slide=50):
        self.dataset = dataset
        self.model = model
        self.window = window
        self.lookback = lookback
        self.slide = slide

    def train(self, maxiter=20):
        trX, trY, _, _ = self.dataset.build(0, 100, 5, 50)
        weights = self.model.weights(trX, seed=1)

        for i in range(maxiter):
            # Update weights with SGD.
            cost = self.model.cost(weights, trX, trY)
            gradient = self.model.grad(weights, trX, trY)
            weights = weights + 1.0*gradient

            print "Iter ", i, ": ", self.model.mean_return(weights, trX, trY)
