import tensorflow as tf
import tensorflow_model_optimization as tfmot

def apply_unstructured_pruning(model, final_sparsity, steps, epochs, config):
  pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=0.0,
          final_sparsity=final_sparsity,
          begin_step=0,
          end_step=steps*(epochs-2)
      )
  }
  
  unstructured_pruned = tfmot.sparsity.keras.prune_low_magnitude(
      model, **pruning_params
  )
  unstructured_pruned.compile(
      optimizer=tf.keras.optimizers.deserialize(config["optimizer"]),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  
  callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
  ]
  
  return unstructured_pruned, callbacks
