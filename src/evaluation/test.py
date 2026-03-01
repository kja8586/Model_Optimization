import numpy as np
import tensorflow as tf

def testing(model, x_test, y_test):
    """Evaluate the model on test data and return accuracy"""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_acc


import numpy as np
import tensorflow as tf


def evaluate_tflite_bytes(tflite_model_bytes, x_test, y_test):

    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    num_samples = x_test.shape[0]
    correct = 0

    for i in range(num_samples):

        input_data = np.expand_dims(x_test[i], axis=0)  # Force batch size = 1

        # Quantize input if needed
        if input_details[0]['dtype'] != np.float32:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(input_details[0]['dtype'])

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize output if needed
        if output_details[0]['dtype'] != np.float32:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        pred = np.argmax(output_data, axis=1)
        true = np.argmax(y_test[i])

        if pred[0] == true:
            correct += 1

    accuracy = correct / num_samples
    print(f"TFLite model accuracy: {accuracy:.4f}")
    return accuracy