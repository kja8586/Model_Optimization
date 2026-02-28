import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def apply_qat(model, lr=5e-5):
# quantize_model wrapper returns model with fake-quant ops (QAT)
  qat_model = tfmot.quantization.keras.quantize_model(model)
  qat_model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )

  return qat_model


def convert_to_tflite_int8(model, representative_data, num_calibration_samples=100):

    def representative_dataset():
        for i in range(min(num_calibration_samples, len(representative_data))):
            sample = representative_data[i : i + 1]
            yield [sample.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    converter._experimental_new_quantizer = True
    converter._experimental_lower_tensor_list_ops = False

    try:
        tflite_model = converter.convert()
    except Exception:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()

    return tflite_model  # ← return bytes instead of saving