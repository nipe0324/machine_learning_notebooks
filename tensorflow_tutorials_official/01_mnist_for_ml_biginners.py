"""

MNIST(0〜9の手書き文字の認識)を行う

参考: https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners

流れ
1. データセットを読み込む
2. 手書き文字認識のモデルを作成する（モデルは、画像のすべてのピクセルを確認する）
3. 数千のトレーニングセットでトレーニングをする
4. テストセットでモデルの精度をチェックする
"""

# 1. データセットを読み込む
# トレーニングセット(mnist.train) 55,000件
# バリデーションセット(mnist.validation) 10,000件
# テストセット(mnist.test) 5,000件
#
# 各データセットには、手書き文字の images, 正解の labels がある(0-9の値)
# (Ex: mnist.train.images, mnist.train.labels)
#
# 各画像は28px x 28pxなので、784 x 1 のベクトルにしてあります。
# ※本来は、ベクトル化すると性能が悪くなるのですが初心者向けのためベクトル化してあります
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist.train.images.shape
# (55000, 784)
# 画像784のベクトルが55,000件ある

mnist.train.labels.shape
# (55000, 10)
# 正解ラベルのone-hot vector 10　が 55,000件ある
# one-hot vectorは0,1で表されて、1の箇所が正解になる(例: [0, 0, 1, ...] => 2, [0, 0, 0, 1, ...] => 3)



# 2. 手書き文字認識のモデルを作成する（モデルは、画像のすべてのピクセルを確認する）
import tensorflow as tf

# トレーニングセットの格納用変数xを作成
# tf.placeholderのNoneの箇所は、tensorflowのトレーニング実行時に値を設定できる
x = tf.placeholder(tf.float32, [None, 784])

# 重みW, バイアスbの変数を0で初期化する
# tf.Variableは変更可能なTensor(n次元行列)を格納できる
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# モデルを作成
# matmul(matrix multiple)は、トレーニングセットx(?x784行列)と重みW(784x10行列)を掛け算する=>?x10行列になる
# その結果にバイアスb(10x1)を足す
# 最後に、softmaxを実行する
# softmax関数は出力を確率に変換する関数です。2が80%、5が10%のような形で出力されます。
y = tf.nn.softmax(tf.matmul(x, W) + b)

# コスト関数(loss, 損失関数とも呼ばれる）を定義する
# コスト関数は、正解とモデルの予測の誤差を数値化することで、良いモデルか悪いモデルかを判断する関数です。
# 今回はCross Entropyというコスト関数を定義します
# Cross Entropy: Hy'(y)= -∑yi'log(yi)
# y: モデルの予測、y': 正解
# 詳細はこちら:http://colah.github.io/posts/2015-09-Visual-Information/
y_ = tf.placeholder(tf.float32, [None, 10]) # y'の格納場所を定義
# Cross Entropyの数式の定義
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# コスト関数を最小化するために、最適化アルゴリズムを選ぶ
# コスト関数を最小化すると、正解とモデルの予測の誤差(=コスト関数)が最小化されるので、より良いモデルが作成されます。
# その最適化アルゴリズムにはいくつかの種類がありますが、今回はGradientDescentというアルゴリズムを使います。
# その他のOPtimizerはこちら:https://www.tensorflow.org/versions/master/api_docs/python/train.html
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 変数を初期化する
init = tf.initialize_all_variables()


# 3. 数千のトレーニングセットでトレーニングをする
# TensorFlowは定義ステップと実行ステップにわかれていて、実行ステップでセッションを作成しないと
# 出力結果がでないという仕組みになっています。これはトレーニングを効率的に実行するためにこうなっています。
# いままでは定義ステップで、ここからが実行ステップです
sess = tf.Session() # 定義したモデルをSessionに展開するためにSessionを取得します。
sess.run(init) # 変数を初期化します
# トレーニングを1000回実行します
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) # MNISTのトレーニングセットから100ずつデータをとってくる
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # とってきたデータを、placeholderのNoneにセット

# 4. テストセットでモデルの精度をチェックする
# 正しい予測を計算する
# argmaxは指定したaxisの最大の値のインデックスを返します
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 精度（正答率）を計算する
# cast関数で、[True, False, True, ...] の配列を [1, 0, 1, ...] の配列に変換し、
# reduce_mean で精度（正答率）を計算します。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 実際に精度を表示します
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100)
# 92.150002718 （92%の精度です）
