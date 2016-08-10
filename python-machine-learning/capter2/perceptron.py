import numpy as np

class Perceptron(object):
  """パーセプトロンの分類器
  パラメータ
  ----------------
  eta : float
    学習率（0.0より大きく1.0以下の値）
  n_iter : int
    トレーニングデータのトレーニング回数
  属性
  ----------------
  w_ : 1次元配列
    適合後の重み
  errors_ : リスト
    各エポックでの誤分類数
  """

  def __init__(self, eta=0.01, n_iter=10):
    self.eta = eta
    self.n_iter = n_iter

  def fit(self, X, y):
    """トレーニングデータに適合させる
    パラメータ
    ---------------
    X : 行列, shape = [n_samples, n_features]
      トレーニングデータ
      n_smaplesはサンプル数、n_featuresは特徴量の個数
    y : ベクトル, shape = [n_samples]
      目的変数
    戻り値
    ---------------
    self : object
    """
    self.w_ = np.zeros(1 + X.shape[1])
    self.errors_ = []
    for _ in range(self.n_iter) # トレーニング回数分学習を実施
      errors = 0
      for xi, target in zip(X, y):
        # 重み w1, ... wmの更新
        # ⊿wj= α(y(i)-y(i))xj(i) (j = 1,..., m)
        update = self.eta * (target - self.predict(xi))
        self.w_[1:] += update * xi

        # 重み w0の更新: ⊿w0 = α(y(i) - y(i))
        self.w_[0] += update
        # 重みの更新が 0 でない場合は誤分類としてカウント
        errors += int(update != 0.0)
      # 反復関数ごとの誤差を格納
      self.errors_.append(errors)
    return self

  def net_input(self, X):
    """総入力を計算"""
    return np.dot(X, self.w_[1:]) + self.w_[0]

  def predict(self, X):
    """1ステップ後のクラスラベルを返す"""
    return np.where(self.net_input(X) >= 0.0, 1, -1)
