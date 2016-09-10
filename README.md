# Machine Learning tutorials & sandbox

# 線形回帰 (Liner Regression)

特徴Xに近似する関数を作成し、その関数により値を予測する

## 重要な概念

* データセット (x, y):
* 仮説関数 hθ(x) = θ0 + θ1x: データxが与えられたときの予測を行うモデル
* パラメータ θ0, θ1: 仮説関数（モデル）のパラメータ
* コストファンクション J(θ): 仮説関数の予想値とトレーニングセットの正解を比較しモデルの質を評価する計算式。もし、0ならトレーニングセットと予測の誤差はないということで、完璧に予測できているということ <= 過学習の可能性ありあり
* Gradient Descent（最急降下法）: minJ(θ)を求めるアルゴリズム

## 流れ

1. データセット（特徴Xとラベルy）を用意する
1.1. 必要なら、feature normalization(feature scalling)（データの正規化）を行う
1.2. トレーニングセット、クロスバリデーション、テストセットと３つに分ける
2. トレーニングセットを使い、仮説関数 hθ(x) のパラメータθを算出する。
2.1. 最適なθを求めるには、コストファンクション J(θ) の最小値を求める（minJ(θ)）
2.2. minJ(θ)を行うには、Gradient Descent Algorithmを使う
3. テストセットでモデル（仮説関数）の精度の評価を行う

## 精度向上

* 分散している（非線形）の場合は、精度は一定以上は上がらないので他のアルゴリズムを使う必要がある。たぶん、ニューラルネットワークなど。
* 2次関数にフィットするようなデータの場合、特徴を増やして2次感数にすることで、精度は向上する。（3次, 4次, ...も同様）
  * TODO: どういうロジックでPolynomial featuresを導入する？
* 外れ値に大きく影響を受けるらしい。トレーニングデータを多くして対応。たぶん。

## 参考

* [Coursera week 1](https://www.coursera.org/learn/machine-learning/home/week/1)
* [Coursera week 2](https://www.coursera.org/learn/machine-learning/home/week/2)


# ロジスティック回帰（Logistic Regression）

特徴Xを区別する関数を作成し、その関数により0/1を判断する。

## 重要な概念

* 仮説関数 hθ(x) = g(z) = g(θT * x): 閾値 0.5など以上なら1(positive), 以下なら0(negative)と判断する。閾値は適宜変えてOK
* シグモイド関数 g(z) = 1 / 1 + e^-z: 0 <= g(z) <= 1 となる関数
* コストファンクション
* GradientDescent

## 参考

* [Coursera week 3](https://www.coursera.org/learn/machine-learning/home/week/3)

# Regularization（正規化）

* オーバーフィッティング（過学習）をしないようにするための手法
* オーバーフィッティングとは、トレーニング時にトレーニングセットにだけフィットしすぎるモデルが作成されてしまい、実運用で精度がでなくなってしまう問題。高次の関数を適用するとオーバーフィッティングになりやすい。でも、データ量を増やせばオーバーフィッティングは徐々に解消される。
* 正規化の重みλを大きくするほどオーバーフィッティングしずらくなる。

## 参考

* [Coursera week 3](https://www.coursera.org/learn/machine-learning/home/week/3)

# 異常検知（Anomaly Detection）

特徴Xに対して、ガウス分布を計算し、確率が閾値εより低い場合に異常と判断する

## 重要な概念

* ガウス分布（正規分布）: 平均値の付近に修正機するようなデータの分布を表した変数に関する確率分布
* 多変量ガウス分布

## アルゴリズムの流れ

1. 異常を検知できる特徴xiを選ぶ
2. パラメータμ、σをフィットさせる
3. 新しいサンプルxを与え、p(x)を計算する
4. p(x) < ε の場合、異常と判断する

## 精度を上げるには

* 異常を検知できる特徴を探す（正規分布している特徴か？）
* εの選び方（正解ラベルをPrecision、Recall、F1スコアを計算し、εを機械的に選ぶこともできる）

## 参考

* [Coursera week 9](https://www.coursera.org/learn/machine-learning/home/week/9)

# 商品推薦（Recommendation）

https://www.coursera.org/learn/machine-learning/home/week/9


## RNN

* RNN(Recurrent Neural Network: 再帰型リカレントニューラルネットワーク)は、ディープラーニングの主要アルゴリズムのひとつ。時間軸をとり、前回の出力を入力とすることで、多層ニューラルネットワークとするモデル。
* LSTM(Long Short Term Memory)は、RNNにゲートという機構を設けることで長距離依存問題を対応したもの。

わかりやすい記事
原文: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
日本語訳: http://qiita.com/KojiOhki/items/89cd7b69a8a6239d67ca

## 精度を上げるには

* forget bias は経験でいじる。1.0はゲートがかなりあいている。
* activationを変えてみる。logistic, tanh, Reluなど
