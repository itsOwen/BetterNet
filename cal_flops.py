import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from metrics import binary_crossentropy_dice_loss

def estimate_flops_and_params(model):

    num_params = model.count_params()

    try:
        from tensorflow.profiler import Profiler
        with Profiler(display_stdout=False) as profiler:
            _ = model(tf.random.normal((1, 224, 224, 3)))
        profile_results = profiler.get_results()
    except ImportError:
        print("Profiler not available, using approximate FLOP estimation.")
        estimated_flops = 2 * num_params
        gflops = estimated_flops / 1e9
        print(f"Estimated GFLOPs (based on model size): {gflops:.4f}")

    num_iterations = 100
    total_inference_time = 0

    for _ in range(num_iterations):
        start_time = time.time()
        _ = model.predict(tf.random.normal((1, 224, 224, 3)))
        end_time = time.time()
        total_inference_time += end_time - start_time

    avg_inference_time = total_inference_time / num_iterations
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")

    return num_params, estimated_flops

if __name__ == "__main__":
    trained_model_path = "model/model.keras"
    loaded_model = load_model(trained_model_path, custom_objects={'binary_crossentropy_dice_loss': binary_crossentropy_dice_loss})  # Pass the custom loss function
    loaded_model.summary()

    num_model_params, estimated_model_flops = estimate_flops_and_params(loaded_model)