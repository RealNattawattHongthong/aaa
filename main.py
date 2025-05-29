import tensorflow as tf
import numpy as np
import time
import threading

# Settings
TEST_DURATION_SECONDS = 2 * 60 * 60  # 2 hours
TENSOR_SIZE = 2048  # large matrix for GPU/CPU stress
NUM_CPU_THREADS = 4  # adjust based on your CPU

# CPU Load Function (Matrix Multiplication)
def cpu_stress():
    print("[CPU] Starting CPU load...")
    a = np.random.rand(TENSOR_SIZE, TENSOR_SIZE)
    b = np.random.rand(TENSOR_SIZE, TENSOR_SIZE)
    end_time = time.time() + TEST_DURATION_SECONDS
    while time.time() < end_time:
        _ = np.dot(a, b)

# GPU Load Function (TensorFlow Matrix Ops)
@tf.function
def gpu_stress_step(a, b):
    return tf.matmul(a, b)

def gpu_stress():
    print("[GPU] Starting GPU load...")
    with tf.device('/GPU:0'):  # fallback to CPU if no GPU
        a = tf.random.normal((TENSOR_SIZE, TENSOR_SIZE))
        b = tf.random.normal((TENSOR_SIZE, TENSOR_SIZE))
        end_time = time.time() + TEST_DURATION_SECONDS
        while time.time() < end_time:
            _ = gpu_stress_step(a, b)

def main():
    print("Starting AI-style CPU/GPU stress test for 2 hours...")

    # Start CPU stress threads
    cpu_threads = []
    for _ in range(NUM_CPU_THREADS):
        t = threading.Thread(target=cpu_stress)
        t.start()
        cpu_threads.append(t)

    # Start GPU stress thread
    gpu_thread = threading.Thread(target=gpu_stress)
    gpu_thread.start()

    # Wait for all threads
    for t in cpu_threads:
        t.join()
    gpu_thread.join()

    print("Load test completed.")

if __name__ == '__main__':
    main()
