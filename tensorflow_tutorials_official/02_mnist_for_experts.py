"""
MNIST(0〜9の手書き文字の認識)を行う
Deep Convolutional MNIST Clasifierを使う(CNNを使う)

参考: https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts

流れ
1. データセットを読み込む

"""

# 1. データセットをを読み込む
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

