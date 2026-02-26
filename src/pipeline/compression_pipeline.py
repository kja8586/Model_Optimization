from src.data.mnist import load_mnist

from src.models.base_model import create_large_model
from src.models.structured import create_pruned_model_structure
from src.models.unstructured import apply_unstructured_pruning

from src.training.trained import train
from src.utils.logging import log_strategy
from src.utils.size import get_model_size_kb, get_gzipped_model_size_kb

def run_pipeline():
  (x_train, y_train), (x_test, y_test) = load_mnist() # Loading dataset
  large_model = create_large_model()
  large_hist = train(large_model, x_train, y_train, epochs=10, batch_size=32)
  log_strategy("STRATEGY 1 - LARGE BASE MODEL")
  
