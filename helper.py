import os
import numpy as np
import gzip
import shutil
import tensorflow as tf

def get_model_size_kb(model_path):
   """Get model size in KB"""
   return os.path.getsize(model_path) / 1024

def get_gzipped_size_kb(model_path):
   """Get gzipped size (real compression)"""
   temp_zip = model_path + '.gz'
   with open(model_path, 'rb') as f_in:
       with gzip.open(temp_zip, 'wb') as f_out:
           shutil.copyfileobj(f_in, f_out)
   size = os.path.getsize(temp_zip) / 1024
   os.remove(temp_zip)
   return size

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

def count_filters(model):
   """Count total filters in conv layers"""
   total_filters = 0
   for layer in model.layers:
       if isinstance(layer, tf.keras.layers.Conv2D):
           total_filters += layer.filters
   return total_filters

def count_parameters(model):
    """Count total trainable parameters"""
    return model.count_params()

def get_model_summary(model):
   """Get detailed model architecture summary"""
   summary = []
   model.summary(print_fn=summary.append)
   return '\n'.join(summary)

def convert_to_tflite_int8(model, x_train, save_path):
     import absl
     absl.logging.set_verbosity(absl.logging.ERROR)
     def representative_dataset():
         for i in range(100):
             yield [x_train[i:i+1]]

     converter = tf.lite.TFLiteConverter.from_keras_model(model)
     converter.optimizations = [tf.lite.Optimize.DEFAULT]
     converter.representative_dataset = representative_dataset
     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
     converter.inference_input_type = tf.int8
     converter.inference_output_type = tf.int8

     converter._experimental_new_quantizer = True
     converter._experimental_lower_tensor_list_ops = False

     try:
         tflite_model = converter.convert()
     except:
         converter.inference_input_type = tf.float32
         converter.inference_output_type = tf.float32
         tflite_model = converter.convert()
     with open(save_path, 'wb') as f:
          f.write(tflite_model)

     return get_model_size_kb(save_path)
def evaluate_tflite(tflite_path, x_test, y_test):
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            correct = 0

            for i in range(len(x_test)):
                input_data = x_test[i:i+1]

                if input_details[0]['dtype'] != np.float32:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    input_data = input_data / input_scale + input_zero_point
                    input_data = input_data.astype(input_details[0]['dtype'])

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])

                if output_details[0]['dtype'] != np.float32:
                    output_scale, output_zero_point = output_details[0]['quantization']
                    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

                if np.argmax(output_data) == np.argmax(y_test[i]):
                    correct += 1

            return correct / len(x_test)

        except Exception as e:
            print("TFLite evaluation warning:", e)
            return 0.0