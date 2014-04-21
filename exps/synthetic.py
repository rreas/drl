from sys import path
path.append('src/')

from sys import path
path.append('src/')

import matplotlib.pyplot as plt
from numpy import append, zeros

from trainer import AdaGradTrainer
from models import Nonlinear, Linear
from dataset import Dataset
from utils import synthetic, wealth, sharpe

window = 100
slide = 50
lookback = 5
delta = 0.0
lmb = 0.1

hidden = 15

model = Linear(delta=delta, lmb=lmb)
#model = Nonlinear(delta=delta, lmb=lmb, hidden=hidden)

series = synthetic(10000, seed=1)
data = Dataset(series, [])
trainer = AdaGradTrainer(data, model)

returns = trainer.train(window=window, lookback=lookback, slide=slide, maxiter=40)
returns = append(zeros(len(series)-len(returns)), returns)

# Linear     : 2,618,987,047.18
# Nonlin (15): 2,272,781,214.71
# Nonlin (10): 2,884,104,773.60
# Nonlin (5) : 1,232,632,146.82

# Linear: 

print "Wealth: ", wealth(returns)[-1]

x_axis = range(len(series))

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(x_axis, series)
axarr[0].set_title('Prices')
axarr[1].plot(x_axis, wealth(returns))
axarr[1].set_title('Wealth')
axarr[2].plot(x_axis, sharpe(returns))
axarr[2].set_title('Sharpe')
axarr[3].plot(x_axis, returns)
axarr[3].set_title('Returns')
plt.show()

