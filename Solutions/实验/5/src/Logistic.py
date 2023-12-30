import numpy as np
class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.gamma = gamma
    
    def sigmoid(self, x):
        """The logistic sigmoid function"""
        return 1.0 / (1 + np.exp(-np.dot(self.w, x)) )
    
    def count_loss(self, X, y):
        """count loss"""
        loss = 0
        for i in range(y.size):
            loss -= y[i] * np.dot(self.w, X[i])
            loss += np.log(1 + np.exp(np.dot(self.w, X[i])))
        if self.penalty == "l1":
            return loss + self.gamma * np.sum(np.abs(self.w))
        return loss + self.gamma * np.sum(self.w ** 2) / 2.0
        
    def count_grad_loss(self, X, y):
        """count gradient of loss"""
        grdloss = np.zeros(X[0].size)
        for i in range(y.size):
            grdloss -= (y[i] - 1 + 1.0/(1 + np.exp(np.dot(self.w, X[i]))))*X[i]
        if self.penalty == "l1":
            return self.gamma * np.sign(self.w) + grdloss
        return self.gamma * self.w + grdloss

    def fit(self, X, y, lr=0.01, tol=1e-4, max_iter=1e3, if_count=False):
        """
        Fit the regression coefficients via gradient descent or other methods 
        """
        if self.fit_intercept == True:
            X = np.c_[X, np.ones(y.size)]
        self.w = np.zeros(X[0].size)
        count = 0
        ls = []
        while np.sum(self.count_grad_loss(X, y) ** 2) > tol:
            if count == max_iter:
                break
            count += 1
            if if_count: ls.append(self.count_loss(X, y))
            self.w = self.w - lr * self.count_grad_loss(X, y)
        return ls


    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        if self.fit_intercept == True:
            X = np.c_[X, np.ones(X.shape[0])]
        y_p = []
        for i in range(X.shape[0]):
            if (self.sigmoid(X[i]) > 0.5):
                y_p.append(1)
            else:
                y_p.append(0)
        return np.array(y_p)