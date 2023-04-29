# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
     # 加载数据
    X_train, y_train = read_data()
    # 添加常数列
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # 岭回归
    def ridge_regression(X, y, alpha):
        beta = np.linalg.inv(X.T @ X + alpha * np.identity(X.shape[1])) @ X.T @ y
        return beta
    # 默认alpha=1
    alpha = 1
    beta = ridge_regression(X_train, y_train, alpha)
    # 预测
    data = np.hstack((1, data))
    prediction = data @ beta
    return prediction
    
def lasso(data):
    # 加载数据
    X_train, y_train = read_data()
    # 添加常数列
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # Lasso回归
    def lasso_regression(X, y, alpha, max_iter=1000, tol=1e-4, eta=0.01):
        beta = np.zeros(X.shape[1])
        for i in range(max_iter):
            beta_new = beta - eta * X.T @ (X @ beta - y) + alpha * np.sign(beta)
            if np.linalg.norm(beta_new - beta) < tol:
                break
            beta = beta_new
        return beta
    # 默认alpha=1
    alpha = 1
    beta = lasso_regression(X_train, y_train, alpha)
    # 预测
    data = np.hstack((1, data))
    prediction = data @ beta
    return prediction

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
