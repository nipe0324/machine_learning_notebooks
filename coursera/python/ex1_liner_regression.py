"""
屋台の利益を予想するシンプルな線形回帰

あなたは、レストランのオーナーでいろいろな街で店を出しています。
各街での人口と利益のデータを持っています。
新しい街にレストランを出すときに、人口から利益を予測してみます。

Source: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データの取得
path = os.getcwd() + '/data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.shape #=> (97, 2)
data.head()
"""
   Population   Profit
0      6.1101  17.5920
1      5.5277   9.1302
2      8.5186  13.6620
3      7.0032  11.8540
4      5.8598   6.8233
"""

# データの詳細を取得（平均(mean)、標準(std)など）
data.describe()
"""
       Population     Profit
count   97.000000  97.000000
mean     8.159800   5.839135
std      3.869884   5.510262
min      5.026900  -2.680700
25%      5.707700   1.986900
50%      6.589400   4.562300
75%      8.578100   7.046700
max     22.203000  24.147000
"""

# データをプロットする
plt.ion() # 対話モードにする
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.waitforbuttonpress()

# コストファンクションの定義
# コストファンクションとは、モデルの予想とトレーニングセットの正解を比較しモデルの質を評価する計算式です
def computeCost(X, y, theta):
    # 1/2m∑(Xθ - y)^2 を計算
    #  X: トレーニングセット
    #  y: ラベル（正解の値）
    #  m: トレーニングセット数
    #  θ: 係数（求めたい値）
    inner = np.power(((X * theta.T) - y), 2) # Tは転置行列に変換(列と行を入れ替える）
    m = len(X)
    return np.sum(inner) / (2 * m)

# トレーニングセットの最初に1の列をいれる
data.insert(0, 'Ones', 1)
data.head()
"""
   Ones  Population   Profit
0     1      6.1101  17.5920
1     1      5.5277   9.1302
2     1      8.5186  13.6620
3     1      7.0032  11.8540
4     1      5.8598   6.8233
"""

# ilocメソッドで、行と列を指定してX(トレーニングセット)とy(正解の値)を設定する
cols = data.shape[1]
X = data.iloc[:, 0:cols-1] # [行番号, 列番号] で、:を指定した場合、すべての行や列となります
y = data.iloc[:, cols-1:cols]

# pandas の data frame を numpy の 行列にする
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

# X, theta, yのサイズを確認します
X.shape, theta.T.shape, y.shape
#=> ((97, 2), (2, 1), (97, 1))

# 現状でコストファンクションを計算してみます
computeCost(X, y, theta)
#=> 32.072733877455676

# GradientDescent(最急降下法)
# θj := θj - α * 1/m ∑(hθ(xi) - yi) * xj
#  α: 学習率（learning rate）
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

# 変数を初期化する
alpha = 0.01 # 学習率（learning rate）
iters = 1000 # イテレーション回数

# GradientDescentを実行する
g, cost = gradientDescent(X, y, theta, alpha, iters)

# 実行した後の係数gでコストファンクションを計算する
computeCost(X, y, g)
#=> 4.5159555030789118

# グラフに表示する
x = np.linspace(data.Population.min(), data.Population.max(), 100)
fig, ax = plt.subplots(figsize=(12, 8))
f = g[0, 0] + (g[0, 1] * x)
ax.plot(x, f, 'r', label="Prediction")
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.waitforbuttonpress()

# グラフに表示2
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.waitforbuttonpress()


# scikit-learnでの実装
# from sklearn import linear_model
# model = linear_model.LinearRegression()
# model.fit(X, y)

# x = np.array(X[:, 1].A1)
# f = model.predict(X).flatten()
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
