import torch
import time

def dgemm_pytorch(A, B):
    """
    Performs matrix multiplication using PyTorch on the GPU and returns the execution time.
    """
    # Ensure tensors are on the same device, although we create them on the device directly.
    # This is good practice if the function were to be used elsewhere.
    device = A.device
    
    # Synchronize before timing to ensure all previous CUDA operations are complete.
    torch.cuda.synchronize(device=device)
    
    start_time = time.time()
    
    # Perform the matrix multiplication
    C = torch.matmul(A, B)
    
    # Synchronize after timing to wait for the matmul operation to finish.
    torch.cuda.synchronize(device=device)
    
    end_time = time.time()
    
    return C, end_time - start_time

if __name__ == '__main__':
    # Check if a CUDA-enabled GPU is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        exit()

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    time_spent = 0.0
    end_time = 1 * 60  # 1 minute in seconds
    n = 512
    multiplication_count = 0

    print(f"Performing matrix multiplications of size {n}x{n} for {end_time} seconds on {device}.")

    try:
        # Generate matrices once before the loop to avoid repeated allocation overhead
        A = torch.randn(n, n, device=device, dtype=torch.float64)
        B = torch.randn(n, n, device=device, dtype=torch.float64)

        while time_spent < end_time:
            # Perform matrix multiplication and get the time
            C, mult_time = dgemm_pytorch(A, B)
            
            time_spent += mult_time
            multiplication_count += 1

        # Clean up tensors to free GPU memory after the loop
        del A, B, C
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"GPU out of memory for n = {n}. Halting.")
        # Clean up memory before breaking
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"\n-------------------------------------------------")
    print(f"Fixed N: {n}")
    print(f"Number of multiplications performed: {multiplication_count}")
    print(f"Total computation time: {time_spent:.2f} seconds")
    print(f"-------------------------------------------------")
