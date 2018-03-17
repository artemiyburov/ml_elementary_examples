from sklearn.utils.estimator_checks import check_estimator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
#from lf import loss_function_x, dloss_function_dx_x

def loss_function_x(x):
    return x*(np.sign(x)+1)/2
		
def dloss_function_dx_x(x):
    return (np.sign(x)+1)/2
'''
def loss_function_x(x):
    return np.exp(x)
		
def dloss_function_dx_x(x):
    return np.exp(x)
'''
class SGClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, nu, la):
        self.nu = nu
        self.la = la

    def fit(self, X, y, cycles):#good idea to run this first
        X, y = check_X_y(X, y)
        train_size = np.shape(X)
        self.X_ = np.c_[X, np.ones(train_size[0])]
        self.y_ = y
        #weights = np.zeros(train_size[1] + 1)
        weights = (1./train_size[0])*np.random.random_sample(
                                     train_size[1] + 1)-.5/train_size[0]
        losses = loss_function_x(np.dot(self.X_, weights)*y)
        Q = np.sum(losses)
        
        fig, ax = plt.subplots()
        for i in np.arange(cycles):
            idx = np.random.choice(np.arange(train_size[0]))
            
            ax.scatter(i,Q)
            
            weights = weights-self.nu*self.X_[idx]*y[idx]*dloss_function_dx_x(
                                                          np.dot(self.X_[idx],weights))
            Q = (1-self.la)*Q + self.la*losses[idx]
            #print(weights)
        ax.set_title('Q(cycles)')
        plt.show()
        self.weights = weights
        return self
        
    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'weights'])

        # Input validation
        X = check_array(X)
        test_size = np.shape(X)
        y = np.sign(np.dot(np.c_[X, np.ones(test_size[0])], self.weights))
        print(self.weights)
        print(np.c_[X, np.ones(test_size[0])])
        return y
