"""
家の大きさと寝室の数から、家の売却額を予想する線形回帰

Source: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-2/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データの取得
path = os.getcwd() + '/data/ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
# data.head()

# feature normalization(feature scalling)
# 家のサイズや寝室の数などスケールが異なるので正規化をする
data = (data - data.mean()) / data.std()
# data.head()

# コストファンクションの定義
def computeCost(X, y, theta):
    # 1/2m∑(Xθ - y)^2 を計算
    #  X: トレーニングセット
    #  y: ラベル（正解の値）
    #  m: トレーニングセット数
    #  θ: 係数（求めたい値）
    inner = np.power(((X * theta.T) - y), 2) # Tは転置行列に変換(列と行を入れ替える）
    m = len(X)
    return np.sum(inner) / (2 * m)


# Gradient Descentの定義
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    m = len(X)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / m) * np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

# Onesカラムを追加
data.insert(0, 'Ones', 1)

# X (トレーニングセット)、y（正解ラベル）を設定
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# 行列(matrice)に変換する
X = np.matrix(X.values)
y = np.matrix(y.values)

# θ（シータ）、α（learning rate）、繰り返し回数を初期化
theta = np.matrix(np.array([0, 0, 0]))
alpha = 0.01
iters = 1000

# トレーニングセットに対して、線形回帰を実行する
g, cost = gradientDescent(X, y, theta, alpha, iters)

# モデルのコストを取得
computeCost(X, y, g)


# プロット
plt.ion() # 対話モードにする
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
