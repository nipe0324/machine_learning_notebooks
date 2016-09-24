import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

def load_iris_data():
    """Irisデータをロードする"""
    # Irisデータセットをロード。人気なデータセットなので、scikit-learnに既に入っている
    iris = datasets.load_iris()
    # 3, 4列目の特徴量を抽出
    X = iris.data[:, [2, 3]]
    # 正解ラベルを取得
    y = iris.target

    # トレーニングデータとテストデータに分割
    # 70%をトレーニングセット、30%をテストセットとしてランダムに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 特徴量のスケーリング
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    # トレーニングデータの平均と標準偏差を計算
    sc.fit(X_train)
    # 計算した平均と標準偏差を用いて標準化
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train, X_test, X_train_std, X_test_std, y_train, y_test


def load_wine_data():
    """Wineデータをロードする"""
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                          header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                       'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    # DataFrameからX, yを取得
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    return X_train, X_test, y_train, y_test


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """2次元のデータセットの決定境界(Decision Regions)をプロットする"""
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                              np.arange(x2_min, x2_max, resolution))

    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 元のグリッドポイントのデータサイズに戻す
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # テストセットを目立たせる
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')

