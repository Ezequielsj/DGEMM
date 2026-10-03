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
- `dgemm_kernels.h` - shared kernels used by the chapter executables, benchmark and validator
- `dgemm_cli.h` - common CLI and portable timing/allocation helpers
- `benchmark_dgemm.c` - deterministic campaign for matrix sizes, blocks and thread counts
- `MKL/` - MKL reference implementation
- `PyTorch_CPU/` and `PyTorch_GPU/` - high-level reference runs
- `run_and_collect.py` - execution and CSV collection script
- `analisar_resultados.py` - summary statistics script
- `validate_dgemm.c` - checks the shared production kernels against a scalar reference
- `resultados_campanha.csv` - raw results from the expanded matched campaign
- `Relatorio_DGEMM.md` - final report
- `resultados_parte1.csv`, `resultados_parte2.csv`, `resultados_parte3.csv` - raw data
- `DGEMM_analise.ipynb` - notebook for exploratory analysis

## Requirements

The original measurements were collected on Windows. The expanded matched campaign can be reproduced on Windows or WSL 2 with GCC and AVX2-capable hardware; the checked-in campaign CSV was collected on Ubuntu 24.04 under WSL 2. The machine should include:

- Python 3
- GCC with support for AVX2 and FMA
- OpenMP support in GCC
- optional MKL installation for the MKL version
- optional PyTorch environment for CPU/GPU versions

## How to run

### Compile the C versions

```bash
make ch2 ch3 ch4 ch5 ch6 benchmark validate
```

### Run one chapter executable

```bash
./Chapter5/program.exe 256 32
./Chapter6/program.exe 256 32 4
```

Arguments are `N [block_size [threads]]`. Blocking sizes must divide `N` and be multiples of 4.

### Run the matched benchmark campaign

```bash
./benchmark_dgemm.exe resultados_campanha.csv 5 1
```

The campaign uses `N=128` and `N=256`, block sizes `8`, `16`, `32`, and `64`, and OpenMP thread counts `1`, `2`, and `4`. It performs one warmup and five measured repetitions per configuration. All variants at a given `N` share deterministic input matrices and monotonic wall-clock timing. The CSV stores the seed, raw times, GFLOPS, maximum absolute error, and validation status.

### Validate and summarize results

```bash
./validate_dgemm.exe
python analisar_resultados.py
```

## Final deliverable

The final report is in [Relatorio_DGEMM.md](Relatorio_DGEMM.md). This file contains the complete academic discussion, including the problem statement, methodology, implementations, experimental results, discussion, and conclusion.

This README is intended as a concise project guide for reproducing and understanding the work.
