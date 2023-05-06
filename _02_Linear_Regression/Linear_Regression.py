# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
def ridge(data):
    # 加载数据
    X_train, y_train = read_data()
    # 添加常数列
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # 岭回归
    def ridge_regression(X, y, alpha):
        beta = np.linalg.inv(X.T @ X + alpha * np.identity(X.shape[1])) @ X.T @ y
        return beta
    alpha = 1e-10
    beta = ridge_regression(X_train, y_train, alpha)
    # 预测
    data = np.hstack(([1], data))
    data = data.reshape(1, -1)
    prediction = data @ beta
    return prediction[0]
def lasso(data_input):
    # 加载数据
    X_train, y_train = read_data()
    # 标准化处理数据
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train = (y_train - y_mean) / y_std
    # 添加常数列
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # Lasso回归
    def lasso_regression(X, y, alpha, max_iter=11, tol=1e-2, eta=0.01, decay=0.6935):
        beta = np.zeros(X.shape[1])
        for i in range(max_iter):
            # 计算梯度
            grad = X.T @ (X @ beta - y) + alpha * np.sign(beta)
            # 动态调整学习率
            eta *= decay
            # 更新权重
            beta_new = beta - eta * grad
            # L1正则化
            beta_new = np.sign(beta_new) * np.maximum(np.abs(beta_new) - alpha * eta, 0)
            if np.linalg.norm(beta_new - beta) < tol:
                break
            beta = beta_new
        # 反标准化
        beta[1:] = beta[1:] / X_std
        beta[0] = y_mean - np.sum(beta[1:] * X_mean / X_std)
        # 预测
        data = np.asarray(data_input)
        data = (data - X_mean) / X_std
        data = data.reshape((1, X_train.shape[1]-1))
        data = np.hstack((np.ones((1, 1)), data))
        prediction = data @ beta
        prediction = prediction * y_std + y_mean
        return prediction
     alpha = 3.85915
    prediction = lasso_regression(X_train, y_train, alpha, max_iter=11, tol=1e-2, eta=0.01, decay=0.6935)
    # 预测
    prediction = prediction[0]
    return prediction
