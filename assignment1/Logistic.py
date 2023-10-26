import numpy as np

class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        ################################################################################
        # TODO:                                                                        #
        # Implement the sigmoid function.
        ################################################################################
        
        return 1.0 / (1 + np.exp(-x) )

        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
    
    def loss(self, X, y):
        loss_val = 0

        # for i in range(y.size):
        #     t = np.dot(self.coef_, X[i])
        #     loss_val += np.log(1+np.exp(t))-y[i]*t
        t = np.dot(X, self.coef_)
        loss_val += np.sum(np.log(1+np.exp(t)))
        loss_val -= np.dot(y, t)

        if self.penalty == "l1":
            return loss_val + self.gamma*np.sum(np.abs(self.coef_))
        else:
            return loss_val + 0.5*self.gamma*np.sum(self.coef_**2)

    def loss_gradient(self, X, y):
        """
        Compute gradient of the loss with respect to w. The gradient of L2 loss is as follows:

        Parameters:
        ----------
        - X: numpy array of shape (n_samples, n_features), input data.
        - y: numpy array of shape (n_samples,), target data.

        Returns
        -------
        - loss_grd: numpy array of shape (n_features,), gradient of the loss with respect to w.
        """


        # loss_grd = np.zeros(X.shape[1])
        # for x_i,y_i in X,y:
        #     loss_grd -= x_i*(1.0/.exp(np.dot(self.w, x_i))*(1+np)+y_i-1)
        t = np.dot(X, self.coef_)
        loss_grd = np.dot(X.T, self.sigmoid(t)-y)
        if self.penalty == "l1":
            return loss_grd + self.gamma*np.sign(self.coef_)
        else:
            return loss_grd + self.gamma*self.coef_


    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent or other methods 
        
        Parameters:
        ----------
        - X: numpy array of shape (n_samples, n_features), input data.
        - y: numpy array of shape (n_samples,), target data.
        - lr: float, learning rate for gradient descent.
        - tol: float, tolerance to decide convergence of gradient descent.
        - max_iter: int, maximum number of iterations for gradient descent.
        Returns:
        -------
        - losses: list, a list of loss values at each iteration.        
        """
        # If fit_intercept is True, add an intercept column
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Initialize coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # List to store loss values at each iteration
        losses = []

        ################################################################################
        # TODO:                                                                        #
        # Implement gradient descent with optional regularization.
        # 1. Compute the gradient 
        # 2. Apply the update rule
        # 3. Check for convergence
        ################################################################################

        err=float('inf')
        count=0
        while count<max_iter or err > tol:
            loss = self.loss(X, y)
            losses.append(loss)
            grad = self.loss_gradient(X, y)
            self.coef_ -= lr*grad
            err = np.linalg.norm(grad)
            count+=1
            if(count%10000==0):
                print("iter: ",count," loss: ",loss," err: ",err)
            if count==max_iter:
                print("Reach max_iter")
                break
        return losses
            
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return losses

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        
        Returns:
        - probs: numpy array of shape (n_samples,), prediction probabilities.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Compute the linear combination of inputs and weights
        linear_output = np.dot(X, self.coef_)
        
        ################################################################################
        # TODO:                                                                        #
        # Task3: Apply the sigmoid function to compute prediction probabilities.
        ################################################################################

        probs = self.sigmoid(linear_output)
        res = np.zeros(probs.shape)
        res[probs>0.5] = 1
        return res
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
