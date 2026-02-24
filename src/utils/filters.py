def count_filters(model):
   """Count total filters in conv layers"""
   total_filters = 0
   for layer in model.layers:
       if isinstance(layer, tf.keras.layers.Conv2D):
           total_filters += layer.filters
   return total_filters
