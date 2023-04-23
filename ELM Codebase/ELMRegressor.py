from master_import import *

class ELMRegressor():
        def __init__(self, L=10, random_state=None):
            self.L = L # Number of hidden neurons
            self.random_state = random_state # random state
            
        def fit(self, X, y=None):
            M = np.size(X, axis=0) # Number of samples
            N = np.size(X, axis=1) # Number of features

            np.random.seed(seed=self.random_state) # set random seed
            
            self.w1 = np.random.uniform(low=-1, high=1, size=(self.L, N+1)) # Weights with bias

            bias = np.ones(M).reshape(-1, 1) # Bias definition
            Xa = np.concatenate((bias, X), axis=1) # Input with bias

            S = Xa.dot(self.w1.T) # Weighted sum of hidden layer
            H = np.tanh(S) # Activation function f(x) = tanh(x), dimension M X L

            bias = np.ones(M).reshape(-1, 1) # Bias definition
            Ha = np.concatenate((bias, H), axis=1) # Activation function with bias

            self.w2 = (np.linalg.pinv(Ha).dot(y)).T # w2' = pinv(Ha)*D
            
            return self
        
        def predict(self, X):
            M = np.size(X, axis=0) # Number of samples
            N = np.size(X, axis=1) # Number of features

            bias = np.ones(M).reshape(-1, 1) # Bias definition
            Xa = np.concatenate((bias, X), axis=1) # Input with bias

            S = Xa.dot(self.w1.T) # Weighted sum of hidden layer
            H = np.tanh(S) # Activation function f(x) = tanh(x), dimension M X L

            bias = np.ones(M).reshape(-1, 1) # Bias definition
            Ha = np.concatenate((bias, H), axis=1) # Activation function with bias

            y_pred = Ha.dot(self.w2.T) # Predictions
            
            return y_pred