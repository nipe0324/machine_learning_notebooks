"""基本的なCNN(Convolutional Neural Network)を構築する"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from libs.utils import *
import matplotlib.pyplot as plt

""" インプットとアウトプットを設定する """
# placeholderを使って後から実際の値を設定できるようにする
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# xは[batch, height*width]の行列なので、4D tensorに変換する
x_tensor = tf.reshape(x, [-1, 28, 28, 1])

""" 1つ目のConvolutional層を作成する """
# Weight は [height x width x input_channels x output_channels]
filter_size = 5
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

# Bias は [output_channels]
b_conv1 = bias_variable([n_filters_1])

# convolution層のグラフを作成
h_conv1 = tf.nn.relu(
    tf.nn.conv2d(input=x_tensor,
                 filter=W_conv1,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv1)

""" 2つ目の層を作成する """
# 1つ目と同様に行う
n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(input=h_conv1,
                 filter=W_conv2,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv2)

# fully-connected層につなぐためにreshapeする
h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * n_filters_2])

""" fully-connected層を作成する """
n_fc = 1024
W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

# 正規化とoverfittingを減らすために dropout を追加する
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

""" softmax層 """
W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

""" 損失関数を作成 """
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

""" 精度の定義 """
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

""" トレーニングの開始 """
print('トレーニング開始')
sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_size = 100
n_epochs = 5
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})
    # クロスバリデーションで精度を計算
    print(sess.run(accuracy,
                   feed_dict={
                       x: mnist.validation.images,
                       y: mnist.validation.labels,
                       keep_prob: 1.0
                   }))

# 学習したkernelを表示
W = sess.run(W_conv1)
plt.imshow(montage(W / np.max(W)), cmap='coolwarm')
plt.show()
