@echo off

:: Set the MKL environment variables
set MKL_INTERFACE_LAYER=LP64
set MKL_THREADING_LAYER=INTEL

:: --- CORRECTED PATHS ---
:: Set the path to the MKL DLLs
set INTEL_MKL_DLL_PATH=C:\Program Files (x86)\Intel\oneAPI\mkl\2025.2\bin

:: !! ADD THIS LINE !!
:: Set the path to the Intel Compiler DLLs (which includes libiomp5md.dll)
:: IMPORTANT: Use the path you discovered in Step 1
set INTEL_COMPILER_DLL_PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.2\bin

:: Add BOTH directories to the PATH for this session
:: The compiler path is added first to ensure its DLLs are found.
set PATH=%INTEL_COMPILER_DLL_PATH%;%INTEL_MKL_DLL_PATH%;%PATH%


:: Run your program
echo --- Running MKL program with INTEL threading ---
program.exe

echo.