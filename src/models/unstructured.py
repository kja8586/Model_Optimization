def apply_unstructured_pruning(model, final_sparsity, steps, epochs, lr=1e-5):
  pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=0.0,
          final_sparsity=final_sparsity,
          begin_step=0,
          end_step=steps*epochs
      )
  }
  
  unstructured_pruned = tfmot.sparsity.keras.prune_low_magnitude(
      model, **pruning_params
  )
  unstructured_pruned.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  
  callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
  ]
  
  return unstructured_pruned, callbacks
