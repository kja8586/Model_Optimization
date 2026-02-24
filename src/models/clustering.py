import tensorflow_model_optimization as tfmot

def apply_clustering(model, n_clusters=8, lr=5e-5):
  
  clustering_params = {
      'number_of_clusters': n_clusters,
      # initialization can be 'linear' or 'kmeans' depending on version
      'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR if hasattr(tfmot.clustering.keras.CentroidInitialization, 'LINEAR') else 'linear',
      'preserve_sparsity': True
  }
  
  clustered_model = tfmot.clustering.keras.cluster_weights(
      model, **clustering_params
  )
  clustered_model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )

  return clustered_model
