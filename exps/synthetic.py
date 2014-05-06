from sys import path
path.append('src/')

import matplotlib.pyplot as plt
from numpy import append, zeros

from trainer import ValidatingTrainer
from models import Nonlinear, Linear
from dataset import Dataset
from utils import synthetic, wealth, sharpe

window = 100
slide = 50
lookback = 5
series = synthetic(501, seed=1)
data = Dataset(series[0:-1], [series[1:]])

models = []
for delta in [0.001]:
    for lmb in [0.0001, 0.001, 0.01]:
        models.append(Linear(delta=delta, lmb=lmb))

trainer = ValidatingTrainer(data, models)

returns, decisions = trainer.train(window=window, lookback=lookback,
        slide=slide, maxiter=100)

padding = zeros(len(series)-len(returns))
returns = append(padding, returns)
decisions = append(padding, decisions)

x_axis = range(len(series))

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(5, sharex=True)
axarr[0].plot(x_axis, series)
axarr[0].set_title('Prices')
axarr[1].plot(x_axis, wealth(returns))
axarr[1].set_title('Wealth')
axarr[2].plot(x_axis, sharpe(returns))
axarr[2].set_title('Sharpe')
axarr[3].plot(x_axis, returns)
axarr[3].set_title('Returns')
axarr[4].plot(x_axis, decisions)
axarr[4].set_title('Decisions')
#plt.show()

