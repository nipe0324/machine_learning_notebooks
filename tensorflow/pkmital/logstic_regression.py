"""１層のニューラルネットワークをつかって、ロジスティック回帰を行う"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

"""データの取得と確認"""

# MNISTデータセットの取得
# https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/download/index.html#dataset-object
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

# MNISTデータセットは、`train`, `validation`, `test`に分かれている
# それぞれ、images, labels, num_examplesでアクセスが可能
print(mnist.train.num_examples,      # 55,000
      mnist.test.num_examples,       # 10,000
      mnist.validation.num_examples) #  5,000

# データセット数 x features tensor （次元の配列)
print(mnist.train.images.shape,  # データセット X (55,000, 784) ※784 = 28*28px
      mnist.train.labels.shape)  # 答え y (55,000, 10) ※10 = 0〜9の数字を表す

# imagesのfeaturesが0〜1に収まっているか
#（featuresの値に偏りがあると学習に時間がかかるため。たぶん。。）
print(np.min(mnist.train.images), # 0.0
      np.max(mnist.train.images)) # 1.0

#  28x28 pxのイメージを１枚表示
plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')
# plt.show() # 表示する


"""TensorFlowによるモデルの作成"""
n_input = 784 # featuresの数(28x28px)
n_output = 10 # 結果の数(0-9の数字に対応)
net_input = tf.placeholder(tf.float32, [None, n_input])

# シンプルな回帰 (y = W * x + b) を記載
W = tf.Variable(tf.zeros([n_input, n_output])) # 784 * 10
b = tf.Variable(tf.zeros([n_output]))          # 10 * 1
net_output = tf.nn.softmax(tf.matmul(net_input, W) + b)

# 正解を格納する、placeholderを作成
y_true = tf.placeholder(tf.float32, [None, 10])

# loss functionを定義
cross_entropy = -tf.reduce_sum(y_true * tf.log(net_output))

# 推定結果(net_output)と正解(y_true)を比較する
correct_predicton = tf.equal(tf.argmax(net_output, 1), tf.argmax(y_true, 1))

# 正しい推定の平均をとる
accuracy = tf.reduce_mean(tf.cast(correct_predicton, "float"))

# Gradient descent(learning rate: 0.01, loss function: cross_entropy)でwをトレーニングする
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

"""TensorFlowで学習"""
# セッションを作成し、変数を初期化
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# トレーニング実施
batch_size = 100
n_epochs = 10
for epock_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={ net_input: batch_xs, y_true: batch_ys })
    # バリデーションデータセットで精度を計算
    print(sess.run(accuracy, feed_dict={ net_input: mnist.validation.images, y_true: mnist.validation.labels }))

# テストデータセットで精度を計算
print(sess.run(accuracy, feed_dict={ net_input: mnist.test.images, y_true: mnist.test.labels }))

