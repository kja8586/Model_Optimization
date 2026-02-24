def get_sparsity(model):
   """Calculate model sparsity"""
   total = 0
   zeros = 0
   for layer in model.layers:
       if hasattr(layer, 'kernel'):
           weights = layer.kernel.numpy()
           total += weights.size
           zeros += np.sum(np.abs(weights) < 1e-7)
   return (zeros / total * 100) if total > 0 else 0
