from numpy import mean, std

def sharpe(returns):
  return mean(returns) / std(returns)
