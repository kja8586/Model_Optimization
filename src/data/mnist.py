import tensotflow as tf

def load_mnist():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  x_train = x_train.astype("float32")/255.0
  x_test = x_test.astype("float32")/255.0

  x_train = x_train.reshape(-1,28,28,1)
  x_test = x_test.reshape(-1,28,28,1)

  y_train = tf.keras.utils.to_categorical(y_train, 10)
  y_test = tf.keras.utils.to_categorical(y_test, 10)

  return (x_train, y_train), (x_test, y_test)
