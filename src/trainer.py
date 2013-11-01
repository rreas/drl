from scipy.optimize import minimize
from numpy import append

from dataset import Dataset
from models import Linear, Nonlinear

class Trainer:

    def __init__(self, raw, model, lookback=10, size=100):
        self.dataset = Dataset(raw)
        self.model = model
        self.lookback = lookback
        self.size = size

    def train(self, maxiter=200):
        returns = []
        weights = None
        previous = 0.

        for x, y in self.dataset.gen(size=self.size, lookback=self.lookback):
            # Make a prediction if we have trained.
            if weights is not None:
                pred = self.model.pred(weights, append(x[0], [previous, 1.]))
                returns.append(self.model.retval(y[0], pred, previous))
                previous = pred

            # Train model on available data
            weights = minimize(self.model.cost,
                               self.model.initial_weights(),
                               (x, y),
                               options={'disp': False, 'maxiter': maxiter}).x
        
        return returns
