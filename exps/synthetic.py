from sys import path
path.append('src/')

import matplotlib.pyplot as plt
from numpy import append, zeros

from trainer import ValidatingTrainer
from models import Nonlinear, Linear
from dataset import Dataset
from utils import synthetic, wealth, sharpe
import plotter

#series = synthetic(4001, seed=1)
#data = Dataset(series[0:-1], [series[1:]])

series = synthetic(4000, seed=1)
data = Dataset(series, [])

#for window in [200]:
#    for slide in [5]:
#        for lookback in [10]:
#            for delta in [0.001]:
#
#                models = []
#                for lmb in [0.0, 0.0001, 0.001, 0.01]:
#                    models.append(Linear(delta=delta, lmb=lmb))
#
#                trainer = ValidatingTrainer(data, models)
#                returns, decisions = trainer.train(window=window, lookback=lookback,
#                        slide=slide, maxiter=20)
#
#                filename = 'figures/synthetic_noside_single_w_%i_s_%i_l_%i_d_%f.pdf' % (
#                        window, slide, lookback, delta)
#                title = 'Single Layer - Synthetic (No Side Information)'
#                print "%s\tWealth: %f\tSharpe: %f" % (filename,
#                        wealth(returns)[-1], sharpe(returns)[-1])
#                plotter.save(filename, title, series, returns, decisions)

for window in [200]:
    for slide in [5]:
        for lookback in [10]:
            for delta in [0.001]:
                models = []

                for hidden in [4, 6, 8]:
                    for lmb in [0.0, 0.0001, 0.001, 0.01]:
                        models.append(Nonlinear(delta=delta, lmb=lmb,
                            hidden=hidden))

                trainer = ValidatingTrainer(data, models)
                returns, decisions = trainer.train(window=window, lookback=lookback,
                        slide=slide, maxiter=20)

                filename = 'figures/synthetic_noside_multi_w_%i_s_%i_l_%i_d_%f.pdf' % (
                        window, slide, lookback, delta)
                title = 'Multiple Layer - Synthetic (No Side Information)'
                print "%s\tWealth: %f\tSharpe: %f" % (filename,
                        wealth(returns)[-1], sharpe(returns)[-1])
                plotter.save(filename, title, series, returns, decisions)
