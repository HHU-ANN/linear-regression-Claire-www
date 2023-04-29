# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

# 岭回归
def ridge(X_test):
    X_train, y_train = read_data()
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    alpha = 1  # 正则化系数
    beta = np.linalg.inv(X_train.T @ X_train + alpha * np.identity(X_train.shape[1])) @ X_train.T @ y_train
    prediction = X_test @ beta
    return prediction
# Lasso回归
def lasso(X_test):
    X_train, y_train = read_data()
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    alpha = 1  # 正则化系数
    max_iter = 1000  # 最大迭代次数
    tol = 1e-4  # 收敛误差
    eta = 0.01  # 初始学习率
    decay = 0.9  # 学习率衰减因子
    beta = np.zeros(X_train.shape[1])
    for i in range(max_iter):
        # 计算梯度
        grad = X_train.T @ (X_train @ beta - y_train) + alpha * np.sign(beta)
        # 动态调整学习率
        eta *= decay
        # 更新权重
        beta_new = beta - eta * grad
        # L1正则化
        beta_new = np.sign(beta_new) * np.maximum(np.abs(beta_new) - alpha * eta, 0)
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    prediction = X_test @ beta
    return prediction

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
