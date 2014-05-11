from sys import path
path.append('src/')

import matplotlib.pyplot as plt
from numpy import append, zeros, array
from utils import synthetic, wealth, sharpe

def save(filename, title, prices, returns, decisions):

    padding = zeros(len(prices)-len(returns))
    returns = append(padding, returns)
    decisions = append(padding, decisions)
    x_axis = range(len(prices))

    # Two subplots, the axes array is 1-d
    nplots = 4
    f, axarr = plt.subplots(nplots, sharex=True)

    axarr[0].plot(x_axis, array(prices)/prices[0])
    axarr[0].set_ylabel('Prices')
    axarr[0].set_title(title)

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
    plt.savefig(filename, bbox_inches='tight')

