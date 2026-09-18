# Makefile for Compiling Matrix Multiplication Examples

# Compiler and common flags
CC = gcc
CFLAGS = -O3 -Wall -Wno-unknown-pragmas

# Specific flags for different versions
AVX_FLAGS = -mavx2 -mfma
OMP_FLAGS = -fopenmp

# MKL specific flags and libraries, assumes oneAPI 2025.2
MKL_INCLUDE = -I"C:/Program Files (x86)/Intel/oneAPI/mkl/2025.2/include"
MKL_LIB_PATH = -L"C:/Program Files (x86)/Intel/oneAPI/mkl/2025.2/lib"
INTEL_COMPILER_LIB_PATH = -L"C:/Program Files (x86)/Intel/oneAPI/compiler/2025.2/lib"
MKL_LIBS = -lmkl_rt -liomp5md -lpthread -m64


# Define all targets
all: ch2 ch3 ch4 ch5 ch6 mkl

# --- Chapter-specific rules ---

# Chapter 2: Basic implementation
ch2: Chapter2/program.exe
Chapter2/program.exe: Chapter2/main_algorithm.c
	$(CC) $(CFLAGS) -o "$@" "$<"

# Chapter 3: AVX implementation
ch3: Chapter3/program.exe
Chapter3/program.exe: Chapter3/main_algorithm.c
	$(CC) $(CFLAGS) $(AVX_FLAGS) -o "$@" "$<"

# Chapter 4: AVX with loop unrolling
ch4: Chapter4/program.exe
Chapter4/program.exe: Chapter4/main_algorithm.c
	$(CC) $(CFLAGS) $(AVX_FLAGS) -o "$@" "$<"

# Chapter 5: AVX with cache blocking
ch5: Chapter5/program.exe
Chapter5/program.exe: Chapter5/main_algorithm.c
	$(CC) $(CFLAGS) $(AVX_FLAGS) -o "$@" "$<"

# Chapter 6: AVX with cache blocking and OpenMP
ch6: Chapter6/program.exe
Chapter6/program.exe: Chapter6/main_algorithm.c
	$(CC) $(CFLAGS) $(AVX_FLAGS) $(OMP_FLAGS) -o "$@" "$<"

# MKL
mkl: MKL/program.exe
MKL/program.exe: MKL/main_algorithm.c
	$(CC) $(CFLAGS) $(MKL_INCLUDE) -o "$@" "$<" $(MKL_LIB_PATH) $(INTEL_COMPILER_LIB_PATH) $(MKL_LIBS)


# --- Housekeeping rules ---

# Clean up all generated executables
clean:
	-del /Q Chapter2\\program.exe
	-del /Q Chapter3\\program.exe
	-del /Q Chapter4\\program.exe
	-del /Q Chapter5\\program.exe
	-del /Q Chapter6\\program.exe
	-del /Q MKL\\program.exe

# Phony targets that don't represent files
.PHONY: all clean ch2 ch3 ch4 ch5 ch6 mkl
