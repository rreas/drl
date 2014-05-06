from sys import path
path.append('src/')

import matplotlib.pyplot as plt
from numpy import append, zeros, array

from trainer import ValidatingTrainer
from models import Nonlinear, Linear
from dataset import Dataset
from utils import synthetic, wealth, sharpe

import quote

mxe = [s[1]['c'] for s in quote.get('MXE', '1990-09-01', '2014-05-01')]
mxf = [s[1]['c'] for s in quote.get('MXF', '1990-09-01', '2014-05-01')]

if len(mxe) != len(mxf):
    raise RuntimeError("Length mismatch in series.")

dataset = Dataset(mxe, [mxf])

for window in [1000]:
    for slide in [200]:
        for lookback in [5]:
            for delta in [0.0, 0.0001, 0.001, 0.01]:

                models = []
                for lmb in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
                    models.append(Linear(delta=delta, lmb=lmb))

                trainer = ValidatingTrainer(dataset, models)
                returns, decisions = trainer.train(window=window, lookback=lookback,
                        slide=slide, maxiter=20)


                padding = zeros(len(mxe)-len(returns))
                returns = append(padding, returns)
                decisions = append(padding, decisions)

                x_axis = range(len(mxe))

                # Two subplots, the axes array is 1-d
                nplots = 4
                f, axarr = plt.subplots(nplots, sharex=True)
                axarr[0].plot(x_axis, array(mxf)/mxf[0], 'r', label='MXF')
                axarr[0].plot(x_axis, array(mxe)/mxe[0], 'b', label='MXE')
                axarr[0].legend(loc=2)
                axarr[0].set_ylabel('Prices')
                axarr[0].set_title('Trading MXE & MXF (Side) with Cost %f' % (delta))

                axarr[1].plot(x_axis, wealth(returns))
                axarr[1].set_ylabel('Wealth')
                #axarr[2].plot(x_axis, sharpe(returns))
                #axarr[2].set_ylabel('Sharpe')
                axarr[2].plot(x_axis, returns)
                axarr[2].set_ylabel('Returns')
                axarr[3].plot(x_axis, decisions)
                axarr[3].set_ylabel('Decisions')

                for i in xrange(nplots):
                    axarr[i].locator_params(axis='y', nbins=6)


                axarr[nplots-1].set_xlabel('Time step (t)')
                plt.draw()
                fn = 'figures/mexico_cost_%f.pdf' % (delta)
                plt.savefig(fn, bbox_inches='tight')
