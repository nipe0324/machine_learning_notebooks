"""
正規化されたロジスティック回帰 (Regularized Logistic Regression)

あなたは、工場のプロダクトマネージャーです。
マイクロチップの２つのテスト結果から"Accept"か"Reject"を判断します。


Source: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/

流れ:
1. データをファイルから取得
2. シグモイド関数、コスト関数、Gradient Descentを定義
3. トレーニングを実施
4. 結果を評価、プロットする
"""

import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

# データを取得
path = os.getcwd() + '/data/ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

# データをプロット
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]
plt.ion()
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')


# プロットすると線形で対応できなさそうなので多項にする
degree = 5
x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)

# データを用意
cols = data.shape[1]
X = np.array(data.iloc[:, 1:cols].values)
y = np.array(data.iloc[:, 0:1].values)
theta = np.zeros(11)
learningRate = 1

# シグモイド関数 g(z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 正規化項付きコスト関数
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    x = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * m) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / m + reg


# 正規化対応されたGradientDescent（1回実行）
def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    return grad


# 予想関数
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# 関数計算
costReg(theta, X, y, learningRate)

# GradientDescent実行
result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, learningRate))
result

# プロット
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
