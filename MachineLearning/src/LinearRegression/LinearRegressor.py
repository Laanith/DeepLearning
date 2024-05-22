import numpy as np
import matplotlib.pyplot as plt


class LinearRegressor:

    def __init__(self, X, y) -> None:
        '''
        Requires these parameters as input : 
        X - an n x d numpy nd-array with n samples and d features.
        y - an n x r numpy nd-array with corresponding target values of r-dimensions
        '''
        self.X : np.ndarray = X
        self.y: np.ndarray = y
        self.weights : np.ndarray = None

    def fit(self):
        temp = self.X.T@self.X
        inv = np.linalg.inv(temp)
        temp_2 = self.X.T@self.y
        self.weights = inv@temp_2

    def predict(self):
        preds = self.X@self.weights
        return preds

    def loss(self):
        return np.mean(np.sum(np.square(self.y - self.predict())))

    def plot(self):
        n = self.X.shape[0]
        plt.scatter(range(n), self.y, label='target')
        plt.scatter(range(n), self.predict(), label='predictions')
        plt.legend()
        plt.show()
        
        
        
