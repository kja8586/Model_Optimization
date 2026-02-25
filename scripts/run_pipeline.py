from src.utils.env import configure_tensorflow
from src.pipeline.compression_pipeline import run_pipeline

def main():
    configure_tensorflow()
    run_pipeline()

if __name__ == "__main__":
    main()
