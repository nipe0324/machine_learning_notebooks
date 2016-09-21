"""パーセプトロン
脳内の1つのニューロンの働き(発火するかしないか)を模倣している

参考: Python機械学習プログラム第2章


* 総入力(net input) z = w1 x1 + ... + wm xm
* 活性化関数(activation function) Φ(z) = {1 (z>=0), -1 (z < 0)

[概要]

バイアス
入力X    = 重みW => 総入力関数 Σ => 活性化関数 ==+==> 出力
                        <=  (重みの更新) =  誤差 」

[アルゴリズムの流れ]
1. 重みを0または小さな乱数で初期化する
2. トレーニングサンプルx(i) ごとに以下の手順を実行する
  a) 出力値y' を計算する ※y'は予想結果
  b) 重みを更新する
    Δwj = η(y-y') xj ※ηは学習率(0.0より大きく0.1以下の定数)
    wj := wj + Δwj

[留意事項]
* パーセプトロンの収束が保証されるのは、
  2つのクラスが線形分離可能であり、学習率が小さい場合に限られる    

[ソースコードの説明]
パーセプトロンクラスを作成する。
fitメソッドでトレーニングを行う。
predictメソッドで予測を行う。
メソッド呼び出し時に作成される属性は、self.w_ のようにアンダースコアをつける
"""

import numpy as np

class Perceptron(object):
    """パーセプトロンの分類器
    パラメータ
    ------------
    eta : float
        学習率 (0.0より大きく1.0以下の値)
    n_iter : int
        トレーニングデータのトレーニング回数
    属性
    -----------
    w_ : 1次元配列
        トレーニング後の重み
    errors_ : リスト
        各エポックでの誤分類数
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """トレーニングデータに適合させる
        パラメータ
        -----------
        X : [n_samples, n_features]
            トレーニングデータ
            n_sampleはサンプルの個数、n_featuresは特徴量の個数
        y : [n_samples]
            目的関数
        戻り値
        ----------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1]) # bias + feature数のinputを0で初期化
        self.errors_ = []
        for _ in range(self.n_iter): # トレーニング回数分トレーニングデータを反復
            errors = 0
            for xi, target in zip(X, y): # 各サンプルで重みを更新
                update = self.eta * (target - self.predict(xi)) # Δw = η(y-y') x
                self.w_[1:] += update # 重みwjの更新 Δwj = η(y-y') xj (j = 1, ..., m)
                self.w_[0] += update # 重みw0の更新 Δw0 = η(y-y')
                errors += int(update != 0.0) # 誤分類の場合カウントする
            self.errors_.append(errors) # 反復回数ごとに誤分類数を詰め込んでおく
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)



