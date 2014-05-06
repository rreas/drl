from sys import path
path.append('src/')

import csv
import matplotlib.pyplot as plt
from numpy import append, zeros

from trainer import AdaGradTrainer
from models import Nonlinear, Linear
from dataset import Dataset
from utils import synthetic, wealth, sharpe

# CSV format -----------------------------------------------------
#  0       1       2       3      4        5         6           |
# ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'SUSPICIOUS']
series = []
with open('data/spy_second.csv', 'r') as f:
    reader = csv.reader(f)
    reader.next()

    for row in reader:
        series.append( (float(row[2]) + float(row[3])) / 2. )

# Close: 1.29226262883
# High:  1.26636438804
# Low:   1.26551192375
# Mean:  1.18798747316

# Using higher cost with nonlinear model.
# Nonlin (high, 0.00005 cost): 1.060636197
# Nonlin (mean, 0.00005 cost): 1.04995628382

window = 500
slide = 50
lookback = 5
delta = 0.00005
lmb = 0.1
hidden = 10

#model = Linear(delta=delta, lmb=lmb)
model = Nonlinear(delta=delta, lmb=lmb, hidden=hidden)

data = Dataset(series, [])
trainer = AdaGradTrainer(data, model)

returns, decisions = trainer.train(
        window=window, lookback=lookback, slide=slide, maxiter=40)
padding = zeros(len(series)-len(returns))
returns = append(padding, returns)
decisions = append(padding, decisions)

# Linear     : 2,618,987,047.18
# Nonlin (15): 2,272,781,214.71
# Nonlin (10): 2,884,104,773.60
# Nonlin (5) : 1,232,632,146.82

# Linear: 

print "Wealth: ", wealth(returns)[-1]

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
plt.show()

