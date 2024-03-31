import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from metrics import binary_crossentropy_dice_loss  # Import the custom loss function

def estimate_flops_and_params(model):

    # Get number of parameters
    num_params = model.count_params()

    # Try using Profiler (if available)
    try:
        from tensorflow.profiler import Profiler
        with Profiler(display_stdout=False) as profiler:
            _ = model(tf.random.normal((1, 224, 224, 3)))  # Run a sample inference
        profile_results = profiler.get_results()
        # ... (Extract and calculate FLOPs from profile data as before)  # Might need adjustment
    except ImportError:
        print("Profiler not available, using approximate FLOP estimation.")
        # Alternative FLOP estimation (e.g., based on model size)
        estimated_flops = 2 * num_params  # Rule of thumb estimation (might not be accurate)
        gflops = estimated_flops / 1e9
        print(f"Estimated GFLOPs (based on model size): {gflops:.4f}")

    # Inference time measurement
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
    # Load the trained model
    trained_model_path = "model/model.keras"
    loaded_model = load_model(trained_model_path, custom_objects={'binary_crossentropy_dice_loss': binary_crossentropy_dice_loss})  # Pass the custom loss function
    loaded_model.summary()

    # Calculate parameters, estimated FLOPs (and GFLOPs), and inference time
    num_model_params, estimated_model_flops = estimate_flops_and_params(loaded_model)