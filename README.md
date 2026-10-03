# DGEMM Performance Investigation

This repository contains the final version of a DGEMM performance analysis project for an undergraduate computer architecture course. The project compares scalar, SIMD, unrolled, blocked and OpenMP implementations of double-precision matrix multiplication and evaluates how different optimization strategies affect execution time and throughput.

## Objective

The main goal is to investigate how software optimizations and architectural features influence the performance of matrix multiplication. The study focuses on the relation between algorithmic structure, memory locality, vector instructions, and thread parallelism.

The central hypothesis is that optimization techniques such as AVX2, FMA, loop unrolling, cache blocking, and OpenMP can significantly improve DGEMM performance, although their impact depends on matrix size, memory access pattern, and overhead introduced by parallel execution.

## Team

- Ezequiel de Jesus Santos - 121056350
- Kelly Pinheiro Soares - 125170716
- Luiza Teixeira Barcellos Rosauro de Almeida - 126423803
- Lucas Pereira Pacheco de Medeiros - 126436084

## Repository structure

- `Chapter1/` - baseline scalar Python version
- `Chapter2/` - baseline scalar C version
- `Chapter3/` - AVX2 implementation
- `Chapter4/` - AVX2 + FMA + unrolling
- `Chapter5/` - blocking / tiling version
- `Chapter6/` - blocking + OpenMP version
- `MKL/` - MKL reference implementation
- `PyTorch_CPU/` and `PyTorch_GPU/` - high-level reference runs
- `run_and_collect.py` - execution and CSV collection script
- `analisar_resultados.py` - summary statistics script
- `validate_dgemm.c` - independent validation of correctness
- `Relatorio_DGEMM.md` - final report
- `resultados_parte1.csv`, `resultados_parte2.csv`, `resultados_parte3.csv` - raw data
- `DGEMM_analise.ipynb` - notebook for exploratory analysis

## Requirements

The project was designed for a Windows environment with GCC and AVX2-capable hardware. For the most complete execution flow, the machine should include:

- Python 3
- GCC with support for AVX2 and FMA
- OpenMP support in GCC
- optional MKL installation for the MKL version
- optional PyTorch environment for CPU/GPU versions

## How to run

### Compile the C versions

```bash
make all
```

Or directly with GCC:

```bash
gcc -O3 -Wall -o Chapter2/program.exe Chapter2/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -o Chapter3/program.exe Chapter3/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -o Chapter4/program.exe Chapter4/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -o Chapter5/program.exe Chapter5/main_algorithm.c
gcc -O3 -Wall -mavx2 -mfma -fopenmp -o Chapter6/program.exe Chapter6/main_algorithm.c
```

### Run the benchmark collection script

```bash
python run_and_collect.py --versions chapter2 chapter3 chapter4 chapter5 chapter6 --num_iterations 5
```

### Reproduce summary statistics

```bash
python analisar_resultados.py
```

### Validate correctness

```bash
gcc -O3 -Wall -mavx2 -mfma -fopenmp -o validate_dgemm.exe validate_dgemm.c
./validate_dgemm.exe
```

## Final deliverable

The final report is in [Relatorio_DGEMM.md](Relatorio_DGEMM.md). This file contains the complete academic discussion, including the problem statement, methodology, implementations, experimental results, discussion, and conclusion.

This README is intended as a concise project guide for reproducing and understanding the work.
