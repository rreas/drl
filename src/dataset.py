from numpy import array, float64, append

class Dataset:
    
    def __init__(self, data, side):
        self.data = array(data, float64)
        self.side = array(side, float64)

    def make(self, data, side, lookback):
        x = []; y = []

        # Scale price series and add in.
        scaled = (data[1:] - data[:-1]) / data[:-1]

        for i in range(0, len(scaled)-lookback+1):
            j = i + lookback

            # Initial example with scaled prices.
            temp_x = scaled[i:j-1]

            # Add in side information relative to prices.
            for s in side:
                temp_x = append(temp_x, (s[i+1:j] - s[i:j-1]) / s[i:j-1])

            # Save the example with bias and label just created.
            x.append(append(1, temp_x))
            y.append(scaled[j-1])

        return x, y

    def build(self, start, window, lookback, slide):
        trD = self.data[start:start+window]
        tsD = self.data[start+window-lookback:start+window+slide]

        if len(self.side) > 0:
            trS = self.side[:,start:start+window]
            tsS = self.side[:,start+window-lookback:start+window+slide]
        else:
            trS = tsS = []

        trX, trY = self.make(trD, trS, lookback)
        tsX, tsY = self.make(tsD, tsS, lookback)

        return array(trX), trY, array(tsX), tsY

    def reset(self, window=100, lookback=10, slide=50):
        self.window = window
        self.lookback = lookback
        self.slide = slide
        self.start = 0

    def can_gen(self):
        return self.start + self.window < len(self.data)

    def gen(self):
        trX, trY, tsX, tsY = self.build(self.start, self.window, self.lookback, self.slide)
        self.start += self.slide
        return trX, trY, tsX, tsY

