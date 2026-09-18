# Matrix Multiplication Performance Analysis

Repository: https://github.com/Ezequielsj/DGEMM

This repository contains different implementations of matrix multiplication, and a Python script to automate the process of running them, collecting performance data, and saving it to a CSV file.

## Integrantes

- Ezequiel de Jesus Santos - 121056350
- Kelly Pinheiro Soares - 125170716
- Luiza Teixeira Barcellos Rosauro de Almeida - 126423803
- Lucas Pereira Pacheco de Medeiros - 126436084

## Overview

The `run_and_collect.py` script is the main tool for this analysis. It can:
- Compile C implementations using `make`.
- Run different versions of the matrix multiplication algorithm (Python, C, MKL, PyTorch).
- Execute each version multiple times.
- Parse the output to extract key performance metrics (matrix size N, number of multiplications, execution time).
- Calculate GFLOPS (GigaFLOPs per second).
- Store the results in a CSV file for further analysis.

## Algorithm Versions

The following algorithm versions are supported:

- `chapter1`: A simple Python implementation.
- `chapter2`: A basic C implementation.
- `chapter3`: C implementation with optimizations.
- `chapter4`: C implementation with further optimizations.
- `chapter5`: C implementation with more optimizations.
- `chapter6`: C implementation with advanced optimizations.
- `mkl`: An implementation using Intel's Math Kernel Library (MKL).
- `pytorch_cpu`: A PyTorch implementation running on the CPU.
- `pytorch_gpu`: A PyTorch implementation running on the GPU.

## Prerequisites

Before running the script, ensure you have the following installed:

- **Python 3**: The script is written in Python.
- **Make**: Required to compile the C versions of the algorithm. Ensure `make` is available in your system's PATH.
- **GCC**: Required by `make` to compile the C source files.
- **Python environments**: The `pytorch_cpu` and `pytorch_gpu` versions require specific Python environments with PyTorch installed. You may need to update the paths to the Python interpreters in the `CONFIG` dictionary within the `run_and_collect.py` script.

## How to Use

You can run the script from your terminal.

### Basic Usage

To run all available algorithm versions with a single iteration each:

```bash
python run_and_collect.py
```

This will create a `performance_results.csv` file with the results.

### Command-Line Arguments

The script accepts the following arguments:

- `--versions`: Specify which versions to run. You can provide one or more version names separated by spaces. If not provided, all versions will be run.
  
  Example:
  ```bash
  python run_and_collect.py --versions chapter1 chapter2 mkl
  ```

- `--num_iterations`: The number of times to run each specified version. The results will be aggregated. Default is 1.

  Example:
  ```bash
  python run_and_collect.py --versions chapter6 --num_iterations 10
  ```

- `--output_csv`: The name of the CSV file to save the results to. Default is `performance_results.csv`.

  Example:
  ```bash
  python run_and_collect.py --output_csv my_results.csv
  ```

### Numerical Validation

The optimized C versions can be checked against an independent scalar reference:

```text
gcc -O3 -Wall -mavx2 -mfma -fopenmp -o validate_dgemm.exe validate_dgemm.c
validate_dgemm.exe
```

The validation uses `N=32`, tolerance `1e-12`, and checks AVX2, FMA/unrolling, blocking, and OpenMP.

### Result Summary Without Jupyter

The CSV statistics can also be reproduced from the terminal:

```text
python analisar_resultados.py
```

## Output CSV Format

The output CSV file will have the following columns:

- `version`: The name of the algorithm version.
- `iteration`: The number of the iteration.
- `n`: The size of the matrix (N x N).
- `multiplications`: The total number of matrix multiplications performed across the iteration.
- `total_time`: The total execution time in seconds for the iteration iterations.
- `gflops`: The calculated GFLOPS (GigaFLOPs per second) based on the iteration data.
- `timestamp`: The date and time when the data was logged.
