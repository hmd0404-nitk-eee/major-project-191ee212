from master_import import *

class DataNormaliser():
    
    def __init__(self, new_range):
        self.range_min = new_range[0] # Min value
        self.range_max = new_range[1] # Max value

    def fit(self, X, y=None):
        self.xmin, self.xmax = X.min(), X.max() # Min and max values of data

        return self

    def transform(self, X):
        X_transform = self.range_min + ( (X-self.xmin)*(self.range_max-self.range_min)/(self.xmax-self.xmin) ) # Normalized data in a new range

        return X_transform

    def inverse_transform(self, X_transform):
        X = self.xmin + ( (X_transform-self.range_min)*(self.xmax-self.xmin)/(self.range_max-self.range_min) ) # Inverse transformation

        return X