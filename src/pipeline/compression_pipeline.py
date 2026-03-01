import yaml
from pathlib import Path
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from src.data.mnist import load_mnist

from src.models.base_model import create_large_model
from src.models.structured import create_pruned_model_structure
from src.models.unstructured import apply_unstructured_pruning
from src.models.clustering import apply_clustering
from src.models.quantization import apply_qat, convert_to_tflite_int8

from src.training.trained import train

from src.evaluation.test import testing, evaluate_tflite_bytes

from src.utils.logging import log_strategy
from src.utils.filters import count_filters
from src.utils.size import get_model_size_kb, get_gzipped_model_size_kb
from src.utils.comparision import compare_strategies

# Load config
ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "configs" / "compression.yaml", "r") as f:
    config = yaml.safe_load(f)  

def run_pipeline():
  (x_train, y_train), (x_test, y_test) = load_mnist() # Loading dataset
  # Step 1: Train a large base model
  print(f"\n{'='*100}")
  print(f"STRATEGY 1 - LARGE BASE MODEL:")
  print(f"{'='*100}")
  large_model = create_large_model(config)
  large_hist = train(large_model, x_train, y_train, epochs=config["base_epochs"], batch_size=config["batch_size"])
  large_acc = testing(large_model, x_test, y_test)
  base_size = get_model_size_kb(large_model)
  base_gzip = get_gzipped_model_size_kb(large_model)
  base_params = large_model.count_params()
  base_filters = count_filters(large_model)
  base_metrics = {
      "accuracy": large_acc,
      "size_kb": base_size,
      "gzip_kb": base_gzip,
      "params": base_params,
      "filters": base_filters}
  log_strategy(large_acc, base_size, base_gzip, base_params, base_filters)

  # Step 2: Structured Pruning
  print(f"\n{'='*100}")
  print(f"STRATEGY 2 - STRUCTURED PRUNING:")
  print(f"{'='*100}")
  structured_model = create_pruned_model_structure(prune_ratio=config["prune_ratio"], config=config)
  structured_hist = train(structured_model, x_train, y_train, epochs=config["structured_epochs"], batch_size=config["batch_size"])
  structured_acc = testing(structured_model, x_test, y_test)
  structured_size = get_model_size_kb(structured_model)
  structured_gzip = get_gzipped_model_size_kb(structured_model)
  structured_params = structured_model.count_params()
  structured_filters = count_filters(structured_model)
  structured_metrics = {
      "accuracy": structured_acc,
      "size_kb": structured_size,
      "gzip_kb": structured_gzip,
      "params": structured_params,
      "filters": structured_filters }
  log_strategy(structured_acc, structured_size, structured_gzip, structured_params, structured_filters)

  # Step 3: Unstructured Pruning
  print(f"\n{'='*100}")
  print(f"STRATEGY 3 - UNSTRUCTURED PRUNING:")
  print(f"{'='*100}")
  unstructured_model, unstructured_callbacks = apply_unstructured_pruning(structured_model, final_sparsity=config["unstructured_sparsity"], steps=len(x_train)//config["batch_size"], epochs=config["base_epochs"], config=config)
  unstructured_hist = train(unstructured_model, x_train, y_train, epochs=config["base_epochs"], batch_size=config["batch_size"], callbacks=unstructured_callbacks)
  unstructured_model = tfmot.sparsity.keras.strip_pruning(unstructured_model)
  unstructured_model.compile(
    optimizer=tf.keras.optimizers.deserialize(config["optimizer"]),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
  unstructured_acc = testing(unstructured_model, x_test, y_test)
  unstructured_size = get_model_size_kb(unstructured_model)
  unstructured_gzip = get_gzipped_model_size_kb(unstructured_model)
  unstructured_params = unstructured_model.count_params()
  unstructured_filters = count_filters(unstructured_model)
  unstructured_metrics = {
      "accuracy": unstructured_acc,
      "size_kb": unstructured_size,
      "gzip_kb": unstructured_gzip,
      "params": unstructured_params,
      "filters": unstructured_filters,
      "sparsity": config["unstructured_sparsity"] }
  log_strategy(unstructured_acc, unstructured_size, unstructured_gzip, unstructured_params, unstructured_filters)

  # Step 4: Clustering
  print(f"\n{'='*100}")
  print(f"STRATEGY 4 - CLUSTERING:")
  print(f"{'='*100}")
  clustered_model = apply_clustering(unstructured_model, n_clusters=config["clusters"])
  clustered_hist = train(clustered_model, x_train, y_train, epochs=config["base_epochs"], batch_size=config["batch_size"])
  clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)
  clustered_model.compile(
      optimizer=tf.keras.optimizers.deserialize(config["optimizer"]),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  clustered_acc = testing(clustered_model, x_test, y_test)
  clustered_size = get_model_size_kb(clustered_model)
  clustered_gzip = get_gzipped_model_size_kb(clustered_model)
  clustered_params = clustered_model.count_params()
  clustered_filters = count_filters(clustered_model)
  clustered_metrics = {
      "accuracy": clustered_acc,
      "size_kb": clustered_size,
      "gzip_kb": clustered_gzip,
      "params": clustered_params, 
      "filters": clustered_filters }
  log_strategy(clustered_acc, clustered_size, clustered_gzip, clustered_params, clustered_filters)

  # Step 5: Quantization
  print(f"\n{'='*100}")
  print(f"STRATEGY 5 - QUANTIZATION:")
  print(f"{'='*100}")
  quantized_model = apply_qat(clustered_model)
  quantized_hist = train(quantized_model, x_train, y_train, epochs=config["qat_epochs"], batch_size=config["batch_size"])
  quantized_acc = testing(quantized_model, x_test, y_test)
  quantized_size = get_model_size_kb(quantized_model)
  quantized_gzip = get_gzipped_model_size_kb(quantized_model)
  quantized_params = quantized_model.count_params()
  quantized_filters = count_filters(quantized_model)
  quantized_metrics = {
      "accuracy": quantized_acc,
      "size_kb": quantized_size,
      "gzip_kb": quantized_gzip,
      "params": quantized_params, 
      "filters": quantized_filters }
  log_strategy(quantized_acc, quantized_size, quantized_gzip, quantized_params, quantized_filters) 

  # Step 6: Convert to TFLite int8
  print(f"\n{'='*100}")
  print(f"STRATEGY 6 - TFLITE INT8:")
  print(f"{'='*100}")
  tflite_model = convert_to_tflite_int8(quantized_model, x_train, num_calibration_samples=100)
  tflite_acc = evaluate_tflite_bytes(tflite_model, x_test, y_test)
  tflite_size = get_model_size_kb(tflite_model)
  tflite_gzip = get_gzipped_model_size_kb(tflite_model)
  tflite_metrics = {
      "accuracy": tflite_acc,
      "size_kb": tflite_size,
      "gzip_kb": tflite_gzip,
      "params": quantized_params,  # params should be same as quantized model
      "filters": quantized_filters } # filters should be same as quantized model
  log_strategy(tflite_acc, tflite_size, tflite_gzip, quantized_params, quantized_filters)

  # Final Comparison
  compare_strategies(base_metrics, structured_metrics, unstructured_metrics, clustered_metrics, quantized_metrics, tflite_metrics)