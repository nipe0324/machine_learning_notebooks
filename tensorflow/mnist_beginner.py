# https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html

# MNIST: 機械学習のHollo, world!のような存在で手書き文字(数字)を判別する
# Softmax regression (累積ロジスチックモデル) :

## データセット の取得
# MNISTデータ: http://yann.lecun.com/exdb/mnist/ にあるが、TensorFlowのコードと一緒にダウンロードさせるようにしている
# データは以下の3つに分かれている
#   1. トレーニング用の55,000件のデータ (mnist.train)
#   2. テスト用の10,000件のデータ (mnist.test)
#   3. 検証用の5,000件のデータ (mnist.validation)
#
# すべてのデータは、手書き数字の"images"とラベルの"labels"がある
# 例: mnist.train.images, mnist.train.labels
#
# イメージは、28 x 28 ピクセル
# 784(28*28)のベクターで、各ピクセルは、0〜1の間の強度(濃淡)をもっている
# ラベルは、イメージの答えとなるような0-9の間の数字


# Learningアルゴリズム => Softmax Regression ###
# 下記の2つのステップを行います
#   1. we add up the evidence of our input being in certain classes.
#   2. we convert that evidence into probabilities.

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

## Implementing the Regression

# TensorFlowライブラリをインポートする
import tensorflow as tf

# "palceholder" はTensorFlowに計算結果を問い合わせるときのインプット値
# "None" は何次元でもよい
x = tf.placeholder(tf.float32, [None, 784])

# "Variable" は変更可能なtensor。計算結果を計算するときに使われる
# W の [784, 10] 784次元のイメージベクターを、10次元のエビデンスベクターにする
# b の [10] はアウトプットに10ずつ加える
W = tf.Variable(tf.zeros([784, 10])) 
b = tf.Variable(tf.zeros([10]))

# モデルを作成する
#   まず、tf.matmul(x, W) x と W を掛け算する (たぶん、matrix multiple の略)
#   その次に、b を足す
#   そして、softmaxを適用させる
y = tf.nn.softmax(tf.matmul(x, W) + b)

## Training

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Gradient descent アルゴリズムでcross_entropyの最小値をもとめる（learning rate 0.5）
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 作成した変数(variables)を初期化する
init = tf.initialize_all_variables()

# Sessionを作成し、変数の初期化をする
sess = tf.Session()
sess.run(init)

# 学習を始める
# データセットから、ランダムな100件のバッチを取得し、train_stepを実行するといった処理をループ
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


## モデルの評価

# argmax(y,1) はモデルが考える正解（ラベル）
# argmax(y_,1)は正しい正解
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# correct_predictionはbooleanの配列なので、数値にキャストして、平均をもとめる
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 精度を出力する => 0.9216
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
