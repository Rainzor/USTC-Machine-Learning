import numpy as np
from Logistic import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from SVM import SVM

def lr_mul(X, y, p, X_t):
    r = LogisticRegression(gamma = 1e-4)
    y1 = y
    for j in range(y1.size):
        if y1[j] in p: y1[j] = 1
        else: y1[j] = 0
    r.fit(X, y1, max_iter=50, lr=1e-7)
    return r.predict(X_t)

def test_lr(X, y, X_t):
    yp1 = lr_mul(X, y, [0,1], X_t)
    yp2 = lr_mul(X, y, [0,2], X_t)
    yp3 = lr_mul(X, y, [0,3], X_t)
    yp4 = lr_mul(X, y, [0], X_t)
    yp5 = lr_mul(X, y, [1], X_t)
    yp6 = lr_mul(X, y, [2], X_t)
    yp7 = lr_mul(X, y, [3], X_t)
    yp = np.zeros(yp1.size, dtype="int")
    x = np.zeros(4, dtype="int")
    for i in range(yp.size):
        now = np.array([yp1[i], yp2[i], yp3[i], yp4[i], yp5[i], yp6[i], yp7[i]])
        x[0] = np.sum(np.array([1,1,1,1,0,0,0]) == now)
        x[1] = np.sum(np.array([1,0,0,0,1,0,0]) == now)
        x[2] = np.sum(np.array([0,1,0,0,0,1,0]) == now)
        x[3] = np.sum(np.array([0,0,1,0,0,0,1]) == now)
        a = np.concatenate(np.argwhere(x == np.max(x)))
        yp[i] = np.random.choice(a)
    return yp

def test_tree(X, y, X_t):
    a = DecisionTreeClassifier(criterion="entropy", max_depth=16, min_samples_leaf=400)
    a.fit(X, y)
    return a.predict(X_t)

def test_mlp(X, y, X_t):
    b = MLPClassifier(hidden_layer_sizes=80, max_iter=150)
    b.fit(X, y)
    return b.predict(X_t)

def test_xgb(X, y, X_t):
    c = XGBClassifier(n_estimators=60, max_depth=15)
    c.fit(X, y)
    return c.predict(X_t)

def svm_mul(X, y, p, X_t):
    r = SVM(X[0].size)
    y1 = y
    for j in range(y1.size):
        if y1[j] in p: y1[j] = 1
        else: y1[j] = -1
    r.fit(X, y1, choose_method="random", iter=50, C=1.5)
    return r.predict(X_t)

def test_svm(X, y, X_t):
    yp1 = svm_mul(X, y, [0,1], X_t)
    yp2 = svm_mul(X, y, [0,2], X_t)
    yp3 = svm_mul(X, y, [0,3], X_t)
    yp4 = svm_mul(X, y, [0], X_t)
    yp5 = svm_mul(X, y, [1], X_t)
    yp6 = svm_mul(X, y, [2], X_t)
    yp7 = svm_mul(X, y, [3], X_t)
    yp = np.zeros(yp1.size, dtype="int")
    x = np.zeros(4, dtype="int")
    for i in range(yp.size):
        now = np.array([yp1[i], yp2[i], yp3[i], yp4[i], yp5[i], yp6[i], yp7[i]])
        x[0] = np.sum(np.array([1,1,1,1,-1,-1,-1]) == now)
        x[1] = np.sum(np.array([1,-1,-1,-1,1,-1,-1]) == now)
        x[2] = np.sum(np.array([-1,1,-1,-1,-1,1,-1]) == now)
        x[3] = np.sum(np.array([-1,-1,1,-1,-1,-1,1]) == now)
        a = np.concatenate(np.argwhere(x == np.max(x)))
        yp[i] = np.random.choice(a)
    return yp