from numpy import array

class Series:
    
    def __init__(self, raw):
        self.raw = array(raw, 'float64')
        self.data = self.raw[1:] / self.raw[:-1]
