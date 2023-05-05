import os
try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
# 岭回归函数
def ridge(data, alpha):
    # 添加截距项
    X = np.column_stack((np.ones(len(data[:, :-1])), data[:, :-1]))
    y = data[:, -1] # 获取预测值
    # 最小二乘法求解
    coef = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + alpha * np.identity(X.shape[1])), X.T), y)
    return coef
# Lasso回归函数
def lasso(data, alpha, lr=0.001, max_iter=10000):
    # 添加截距项
    X = np.column_stack((np.ones(len(data[:, :-1])), data[:, :-1]))
    y = data[:, -1] # 获取预测值
    w = np.zeros(X.shape[1]) # 初始化权重
    # 迭代更新权重
    for i in range(max_iter):
        # 计算梯度
        grad = np.dot(X.T, np.dot(X, w) - y) + alpha * np.sign(w)
        # 更新权重
        w -= lr * grad
    return w
# 读取数据函数
def read_data(path='./data/exp02/'):
    X_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    return np.column_stack((X_train, y_train)) # 将属性和预测值合并为一个数组
data = read_data() # 加载数据
# 岭回归求解
coef_ridge = ridge(data, alpha=0.1)
print(coef_ridge)
# Lasso回归求解
coef_lasso = lasso(data, alpha=0.1)
print(coef_lasso)
