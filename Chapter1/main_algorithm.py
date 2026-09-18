import random
import time

def generate_matrix(n):
    return [[random.randint(-100, 100) for _ in range(n)] for _ in range(n)]


def matrix_multiply(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    start_mult_time = time.time()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    end_mult_time = time.time()
    return C, end_mult_time - start_mult_time

if __name__ == '__main__':
    time_spent = 0
    end_time = 1 * 60  # 1 minute in seconds
    n = 512
    multiplication_count = 0

    print(f"Performing matrix multiplications of size {n}x{n} for {end_time} seconds.")

    try:
        # Generate matrices once before the loop
        A = generate_matrix(n)
        B = generate_matrix(n)

        while time_spent < end_time:
            _, mult_time = matrix_multiply(A, B)
            time_spent += mult_time
            multiplication_count += 1
            # To ensure the total time does not significantly overshoot, check before printing
            if time_spent < end_time:
                print(f"Multiplication {multiplication_count} completed. Total time: {time_spent:.2f}s")

    except MemoryError:
        print(f"Memory allocation failed for n = {n}")

    print("\n-------------------------------------------------")
    print(f"Fixed N: {n}")
    print(f"Number of multiplications performed: {multiplication_count}")
    print(f"Total computation time: {time_spent:.2f} seconds")
    print("-------------------------------------------------")