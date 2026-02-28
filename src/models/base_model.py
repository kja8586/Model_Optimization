import tensorflow as tf

def create_large_model(config):
  model = tf.keras.Sequential([
        # First block - 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second block - 128 filters
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Dense layers - 512 and 256 units
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
  model.compile(
      optimizer=tf.keras.optimizers.deserialize(config["optimizer"]),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )

  return model
