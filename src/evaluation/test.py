import tensorflow as tf
import numpy as np

def testing(model, x_test, y_test):
    """Evaluate the model on test data and return accuracy"""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_acc


def evaluate_tflite_bytes(tflite_model_bytes, x_test, y_test, batch_size=32):
    
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        num_samples = x_test.shape[0]
        correct = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_data = x_test[start:end]
            batch_labels = y_test[start:end]

            # Quantize input if needed
            if input_details[0]['dtype'] != np.float32:
                input_scale, input_zero_point = input_details[0]['quantization']
                batch_data = batch_data / input_scale + input_zero_point
                batch_data = batch_data.astype(input_details[0]['dtype'])

            interpreter.set_tensor(input_details[0]['index'], batch_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Dequantize output if needed
            if output_details[0]['dtype'] != np.float32:
                output_scale, output_zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

            pred = np.argmax(output_data, axis=1)
            true = np.argmax(batch_labels, axis=1)
            correct += np.sum(pred == true)

        accuracy = correct / num_samples
        print(f"TFLite model accuracy: {accuracy:.4f}")
        return accuracy

    except Exception as e:
        print("TFLite evaluation warning:", e)
        return 0.0