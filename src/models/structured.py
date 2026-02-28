import tensorflow as tf

def create_pruned_model_structure(prune_ratio, config):
    """Create a smaller model structure with fewer filters"""
    # prune_ratio here is fraction of filters to remove
    f1 = max(1, int(64 * (1 - prune_ratio)))
    f2 = max(1, int(128 * (1 - prune_ratio)))
    # ensure at least 1 filter
    structured_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(f2, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(f2, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),  # Reduced dense
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    structured_model.compile(
      optimizer=tf.keras.optimizers.deserialize(config["optimizer"]),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
    return structured_model
