import torch
import time

def dgemm_pytorch_cpu(A, B):
    """
    Performs matrix multiplication using PyTorch on the CPU and returns the execution time.
    """
    start_time = time.time()
    
    # Perform the matrix multiplication
    C = torch.matmul(A, B)
    
    end_time = time.time()
    
    return C, end_time - start_time

if __name__ == '__main__':
    device = torch.device("cpu")
    print(f"Using device: {device}")

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
            C, mult_time = dgemm_pytorch_cpu(A, B)
            
            time_spent += mult_time
            multiplication_count += 1
            

        # Clean up tensors to free memory
        del A, B, C

    except MemoryError:
        print(f"CPU out of memory for n = {n}. Halting.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"\n-------------------------------------------------")
    print(f"Fixed N: {n}")
    print(f"Number of multiplications performed: {multiplication_count}")
    print(f"Total computation time: {time_spent:.2f} seconds")
    print(f"-------------------------------------------------")
