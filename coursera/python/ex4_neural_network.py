"""
ニューラルネットワーク（バックプロパゲーション）
手書き文字認識

Source: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-5/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder

data = loadmat('data/ex3data1.mat')
data
"""
{'y': array([[10],
       [10],
       [10],
       ...,
       [ 9],
       [ 9],
       [ 9]], dtype=uint8),
 'X': array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]])}
"""

X = data['X']
y = data['y']
X.shape, y.shape
# ((5000, 400), (5000, 1))

# One-hot encoding（ラベルnをベクトルにしインデックスnをhot(1)にする）
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
y_onehot.shape
# (5000, 10)

y[0], y_onehot[0, :]
# (array([10], dtype=uint8), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]))


# シグモイド関数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# フォワードプロパゲーション
# 現在のパラメータθを使って予想値hを計算する
# ニューラルネットワークは、インプット層 400+bias, 隠れ層 25+bias、出力層 10(0〜9)
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

# コスト関数
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # arrayをmatrixに変換
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # フィードフォワードを実行
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # コストを計算
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    # 正規化項を追加
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    return J


# 変数の初期化
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
theta1.shape, theta2.shape
# ((25, 401), (10, 26))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
a1.shape, z2.shape, a2.shape, z3.shape, h.shape
# ((5000, 401), (5000, 25), (5000, 26), (5000, 10), (5000, 10))

cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

# バックプロパゲーションを実装
def sigmoid_gradient(z):
    """シグモイド関数のgradientを計算する"""
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    # コスト関数と同じ
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    # ここから違う
    # バックプロパゲーションを実行
    for t in range(m):
        alt = a1[t, :] # (1, 401)
        z2t = z2[t, :] # (1, 25)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :]   # (1, 10)
        yt = y[t, :]   # (1, 10)
        d3t = ht - yt  # (1, 10)
        z2t = np.insert(z2t, 0, values=np.ones(1)) # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t)) # (1, 26)
        delta1 = delta1 + (d2t[:, 1:]).T * alt
        delta2 = delta2 + d3t.T * a2t
    delta1 = delta1 / m
    delta2 = delta2 / m
    # Gradient Regularization Termを追加
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return J, grad

J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
J, grad.shape
# (7.1032990365055717, (10285,))

# minimize
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
fmin
"""
     fun: 0.34489741435516924
     jac: array([  2.90591216e-05,  -7.66414145e-07,  -1.05516991e-06, ...,
         1.39116027e-05,   1.19156235e-05,   1.50571185e-04])
 message: 'Max. number of function evaluations reached'
    nfev: 250
     nit: 21
  status: 3
 success: False
       x: array([-0.60421731, -0.00383207, -0.00527585, ...,  3.62867089,
       -0.82906497, -0.41711456])
"""

# 予測値を計算する
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)
y_pred
"""
array([[10],
       [10],
       [10],
       ...,
       [ 9],
       [ 9],
       [ 9]])
"""

# 精度の計算
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100)) # 99.22%
