"""
異常検知 Anomaly Detection
ガウシアンモデルを使って、ネットワーク上のサーバがFailしているのを発見する

Source: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-8/

アルゴリズムの流れ
1. 異常を検知できる特徴xiを選ぶ
2. パラメータμ、σをフィットさせる
3. 新しいサンプルxを与え、p(x)を計算する
4. p(x) < ε ければ、異常と判断する
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats

data = loadmat('data/ex8data1.mat')
X = data['X']
X.shape # (307, 2) 2つの特徴がある307件のデータ

# プロットする
plt.ion()
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1]) # １つのクラスターとなっている

# ガウシアンモデル
def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma

# モデルパラメータμ, σを取得する
mu, sigma = estimate_gaussian(X)
mu, sigma # (array([ 14.11222578,  14.99771051]), array([ 1.83263141,  1.70974533]))

# 閾値εを決めるために、ラベル付けされたバリデーションデータを使ってテスト行う
Xval = data['Xval']
yval = data['yval']

dist = stats.norm(mu[0], sigma[0])
dist.pdf(X[:, 0])[0:50]

p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])

pval = np.zeros((X.shape[0], X.shape[1]))
pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])

def select_threshold(pval, yval):
    """最適な閾値εを探す。F1スコアを使って検証する"""
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    step = (pval.max() - pval.min()) / 1000
    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        # f1値がベストなf1よりも良い場合、ベストなεも更新
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)
epsilon, f1 # (0.0095667060059568421, 0.7142857142857143)

# εを適用する
outliers = np.where(p < epsilon)

# 結果をプロットする
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
