import numpy as np

class SVM:
    def __init__(self, dim):
        self.w = np.zeros(dim)
        self.b = 0.0
    
    def drop_error(self, X, y, q):
        num = y.size
        dim = X[0].size
        drop = []
        for i in range(num):
            max_same = 0
            max_diff = 0
            for j in np.random.choice(num, q):
                if j != i:
                    temp = np.sum(np.abs(X[i] - X[j]))
                    if y[j] == y[i]:
                        if temp > max_same:
                            max_same = temp
                    else:
                        if temp > max_diff:
                            max_diff = temp
            if max_diff < max_same:
                drop.append(i)
        X = np.delete(X, drop, axis = 0)
        y = np.delete(y, drop)
        return X, y

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        y_p = []
        for i in range(X.shape[0]):
            if (np.dot(X[i],self.w) + self.b > 0):
                y_p.append(1)
            else:
                y_p.append(-1)
        return np.array(y_p)

    def construct_predict(self, X, y, lam, C):
        num = y.size
        u = np.zeros(num)
        for i in range(num):
            u[i] = np.dot(self.w, X[i])
        
        count = 0
        self.b = 0
        for i in range(num):
            if lam[i] > 0 and lam[i] < C:
                self.b += y[i] - u[i]
                count += 1
        if count:
            self.b /= float(count)
        
        return u + self.b

    def count_loss(self, u, y, lam):
        loss = np.dot(self.w, self.w)/2.0
        for i in range(y.size):
            loss += lam[i] * (1 - y[i] * u[i])
        return loss

    def fit(self, X, y, C=2.0, choose_method="turning", iter=1e3, if_drop=False, if_draw=False):
        """
        Fit the coefficients via your methods
        """
        if choose_method not in ["turning", "random", "maxerr"]:
            print("choose method error")
            return

        if (if_drop):
            X, y =  self.drop_error(X, y, 100)

        loss = []
        num = y.size
        lam = np.zeros(num)
        
        pos = 0
        neg = 0
        for i in range(num):
            if y[i] == 1:
                pos += 1
            else:
                neg += 1
        
        for i in range(num):
            if y[i] == 1:
                lam[i] = C / float(pos)
            else:
                lam[i] = C / float(neg)
        
        a1 = 0

        self.w = np.zeros(X[0].size)
        for i in range(num):
            self.w = self.w + lam[i] * y[i] * X[i]

        for _ in range(int(iter)):
            u = self.construct_predict(X, y, lam, C)
            if if_draw: loss.append(self.count_loss(u, y, lam))  
            
            if (choose_method == "maxerr"):
                m = 0
                a1 = -1
                for j in range(num):
                    if lam[a1] > 0 and lam[a1] < C:
                        temp = abs(u[j]*y[j] - 1)
                    elif lam[a1] == 0:
                        temp = 1 - u[j]*y[j]
                    else:
                        temp = u[j]*y[j] - 1
                    if temp > m:
                        m = temp
                        a1 = j
                if a1 == -1:
                    return loss
            else:
                count = 0
                if (choose_method == "turning"):
                        a1 = (a1 + 1) % num
                else:
                    a1 = np.random.randint(num)
                while (lam[a1] > 0 and lam[a1] < C and abs(u[a1]*y[a1] - 1) < 1e-6)\
                    or (lam[a1] == 0 and u[a1]*y[a1] >= 1)\
                    or (lam[a1] == C and u[a1]*y[a1] <= 1):
                    if (choose_method == "turning"):
                        a1 = (a1 + 1) % num
                    else:
                        a1 = np.random.randint(num)
                    count += 1
                    if count == num:
                        return loss

            e1 = u[a1] - y[a1]
            
            m = 0
            a2 = -1
            for j in range(num):    
                e2 = u[j] - y[j]
                de = e1 - e2
                if abs(de) > abs(m):
                    a2 = j
                    m = de
            if (a2 == -1):
                return loss

            sum = lam[a1] * y[a1] + lam[a2] * y[a2]
            lam2new = lam[a2] + y[a2] * m / np.dot(X[a1]-X[a2],X[a1]-X[a2])

            if y[a1] == y[a2]:
                l = max(0, lam[a2] + lam[a1] - C)
                h = min(lam[a1] + lam[a2], C)
            else:
                l = max(0, lam[a2] - lam[a1])
                h = min(C + lam[a2] - lam[a1], C)

            self.w = self.w - lam[a2] * y[a2] * X[a2] - lam[a1] * y[a1] * X[a1]
            
            lam[a2] = min(max(lam2new, l), h)
            lam[a1] = (sum - lam[a2] * y[a2]) * y[a1]

            self.w = self.w + lam[a2] * y[a2] * X[a2] + lam[a1] * y[a1] * X[a1]

        return loss