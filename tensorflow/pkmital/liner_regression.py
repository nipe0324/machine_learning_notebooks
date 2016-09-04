"""１個のfactorとbiasで線形回帰を行う"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# データを作成する
n_observations = 100 # featuresの個数
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

# グラフに描く
plt.ion()
flg, ax = plt.subplots(1, 1)
ax.scatter(xs, ys)
flg.show()
plt.draw()

# Xは入力用placeholder, Yは出力用placeholder
# placeholderはsession内で計算するときに値を設定できる変数
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# min_(X, b) || (X*w + b) - y||^2　を最適化する
W = tf.Variable(tf.random_normal([1]), name='weight') # tf.random_normal ランダム値を作成([1] => 1x1行列で)
b = tf.Variable(tf.random_normal([1]), name='bias')
Y_pred = tf.add(tf.mul(X, W), b) # X*W + b = Y(推定結果)

# observationsとpredictionsの距離とそれらの平均をを計測する損失関数を定義
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

# 正規化を行いたい場合は、正規化の項を足す
# cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))

# Wとbを最適化するためにGradientDescentを使う
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# sessionを作成し、グラフを使う
n_epochs = 1000
with tf.Session() as sess:
    # 変数を初期化する
    sess.run(tf.initialize_all_variables())

    # すべてのトレーニングデータを適用させる
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
        print(training_cost)

        if epoch_i % 20 == 0:
            ax.plot(xs, Y_pred.eval(
                feed_dict={X: xs}, session=sess),
                    'k', alpha=epoch_i / n_epochs)
            flg.show()
            plt.draw()

        # global minimumに到着したらトレーニングを終了させる
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost

# グラフを表示
flg.show()
plt.waitforbuttonpress()
