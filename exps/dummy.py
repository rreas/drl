from sys import path
path.append('src/')

from trainer import Trainer
from models import Nonlinear

size = 5
lookback = 2
hidden = 10
delta = 0.0
lmb = 0.1

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
model = Nonlinear(hidden=hidden, lookback=lookback, delta=delta, lmb=lmb)

trainer = Trainer(data, size=size, lookback=lookback, model=model)
print trainer.train(maxiter=20)
