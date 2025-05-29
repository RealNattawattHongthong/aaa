import torch
import time
import threading

# Settings
TEST_DURATION_SECONDS = 2 * 60 * 60  # 2 hours
TENSOR_SIZE = 2048
NUM_CPU_THREADS = 4

# CPU Stress
def cpu_stress():
    print("[CPU] Starting stress test...")
    end_time = time.time() + TEST_DURATION_SECONDS
    a = torch.rand((TENSOR_SIZE, TENSOR_SIZE))
    b = torch.rand((TENSOR_SIZE, TENSOR_SIZE))
    while time.time() < end_time:
        _ = torch.mm(a, b)

# GPU Stress (if available)
def gpu_stress():
    if not torch.cuda.is_available():
        print("[GPU] No GPU found. Skipping GPU stress.")
        return
    print("[GPU] Starting stress test on:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
    a = torch.rand((TENSOR_SIZE, TENSOR_SIZE), device=device)
    b = torch.rand((TENSOR_SIZE, TENSOR_SIZE), device=device)
    end_time = time.time() + TEST_DURATION_SECONDS
    while time.time() < end_time:
        _ = torch.mm(a, b)

def main():
    print("Launching AI-style CPU + GPU load test for 2 hours...")
    
    # Start CPU threads
    cpu_threads = []
    for _ in range(NUM_CPU_THREADS):
        t = threading.Thread(target=cpu_stress)
        t.start()
        cpu_threads.append(t)

    # Start GPU thread
    gpu_thread = threading.Thread(target=gpu_stress)
    gpu_thread.start()

    # Wait for all
    for t in cpu_threads:
        t.join()
    gpu_thread.join()

    print("✅ Load test completed.")

if __name__ == '__main__':
    main()
