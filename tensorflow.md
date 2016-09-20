
## Serving

### Documents

* [Bazel Install with Homebrew](https://www.bazel.io/versions/master/docs/install.html#using-homebrew)
* [Installation](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md)
* [Basic](https://tensorflow.github.io/serving/serving_basic)

### Basic

* モデルのエクスポート

```
bazel build //tensorflow_serving/example:mnist_export
bazel-bin/tensorflow_serving/example/mnist_export /tmp/mnist_model
Training model...

...

Done training!
Exporting trained model to /tmp/mnist_model
Done exporting!
```

* エクスポートしたモデルを読み込み、スタンドアローンのサーバーを起動

```
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/
```

* クライアントから接続しサーバーをテスト

```
bazel build //tensorflow_serving/example:mnist_client
bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000
...
# 1,000件のテスト画像を送信し、10.5%の予想エラー率だった
Inference error rate: 10.5%
```
