import io
import gzip
import tempfile
import tensorflow as tf


def _serialize_model_to_bytes(model):
    # If already serialized (e.g., TFLite model)
    if isinstance(model, (bytes, bytearray)):
        return model

    # Otherwise assume it's a Keras model
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.keras")
        model.save(path)

        with open(path, "rb") as f:
            return f.read()


def get_model_size_kb(model):
    model_bytes = _serialize_model_to_bytes(model)
    return len(model_bytes) / 1024


def get_gzipped_model_size_kb(model):
    model_bytes = _serialize_model_to_bytes(model)

    gz_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buffer, mode="wb") as gz_file:
        gz_file.write(model_bytes)

    return gz_buffer.getbuffer().nbytes / 1024
