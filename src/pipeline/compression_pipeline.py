from src.data.mnist import load_mnist
from src.models.base_model import create_large_model
from src.training.trained import train

def run_pipeline():
  (x_train, y_train), (x_test, y_test) = load_mnist() # Loading dataset
  large_model = create_large_model()
  large_hist = train(large_model, x_train, y_train, epochs=10, batch_size=32)
  
  
