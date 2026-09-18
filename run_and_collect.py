import subprocess
import argparse
import re
import csv
import os
import sys
from datetime import datetime

# Define the configurations for each algorithm version
# --- FIXES APPLIED TO 'path' ENTRIES ---
CONFIG = {
    "chapter1": {
        "path": "Chapter1/main_algorithm.py",
        "type": "python",
        "interpreter": sys.executable,
        "requires_compile": False,
    },
    "chapter2": {
        "path": "Chapter2/program.exe",  # CHANGED: from main_algorithm.exe
        "type": "c",
        "requires_compile": True,
    },
    "chapter3": {
        "path": "Chapter3/program.exe",  # CHANGED: from main_algorithm.exe
        "type": "c",
        "requires_compile": True,
    },
    "chapter4": {
        "path": "Chapter4/program.exe",  # CHANGED: from main_algorithm.exe
        "type": "c",
        "requires_compile": True,
    },
    "chapter5": {
        "path": "Chapter5/program.exe",  # CHANGED: from main_algorithm.exe
        "type": "c",
        "requires_compile": True,
    },
    "chapter6": {
        "path": "Chapter6/program.exe",  # CHANGED: from main_algorithm.exe
        "type": "c",
        "requires_compile": True,
    },
    "mkl": {
        "path": "MKL/run.bat",
        "type": "bat",
        "requires_compile": True, # 'make mkl' handles this
    },
    "pytorch_cpu": {
        "path": "PyTorch_CPU/main_algorithm.py",
        "type": "python",
        "interpreter": r"C:\Users\breno\Documentos\Faculdade\ProjetoIC\SunFlower-main_bak\venv\Scripts\python.exe",
        "requires_compile": False,
    },
    "pytorch_gpu": {
        "path": "PyTorch_GPU/main_algorithm.py",  # CHANGED: from PyTorch/
        "type": "python",
        "interpreter": r"C:\Users\breno\Documentos\Faculdade\ProjetoIC\SunFlower-main_bak\venv\Scripts\python.exe",
        "requires_compile": False,
    },
}

def compile_code(version):
    """Compiles the C code using make if the executable is missing."""
    config = CONFIG[version]
    # Use os.path.join for robust path creation
    executable_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config["path"])
    
    # For C code, the compilation target is the directory name, which should produce 'program.exe'
    if config["type"] == "c":
        # Check for the existence of the actual executable, not the .c file path
        if config["requires_compile"] and not os.path.exists(executable_path):
            print(f"--- Compiling {version}... ---")
            try:
                # The 'make' target is the version name (e.g., 'chapter2')
                target = version
                compile_process = subprocess.run(
                    ["make", target], 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                print(f"--- Compilation of {version} successful. ---")
                print(compile_process.stdout)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"--- ERROR: Compilation of {version} failed. ---")
                if isinstance(e, FileNotFoundError):
                    print("'make' command not found. Please ensure make is installed and in your system's PATH.")
                else:
                    print(e.stderr)
                return False
    return True

def run_and_parse(version):
    """Runs the specified algorithm a single time and parses its output."""
    config = CONFIG[version]
    
    command = []
    run_cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Use os.path.join to create a full, OS-agnostic path to the executable/script
    executable_path = os.path.join(run_cwd, config["path"])

    if config["type"] == "python":
        command = [config["interpreter"], executable_path]
    elif config["type"] == "c":
        command = [executable_path]
    elif config["type"] == "bat":
        command = [executable_path]
        run_cwd = os.path.dirname(executable_path)
    else:
        raise ValueError(f"Unknown type {config['type']} for version {version}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=run_cwd)
        output = result.stdout
        print(output)

        n_match = re.search(r"Fixed N: (\d+)", output)
        mult_match = re.search(r"Number of multiplications performed: (\d+)", output)
        # --- FIX: Made the regex specific to avoid greedy matching across lines ---
        time_match = re.search(r"Total computation time: ([\d.]+) seconds", output)

        if n_match and mult_match and time_match:
            n = int(n_match.group(1))
            mults = int(mult_match.group(1))
            time = float(time_match.group(1))
            
            gflops = (2 * (n**3) * mults) / (time * 1e9) if time > 0 else 0
            
            return {
                "version": version,
                "n": n,
                "multiplications": mults,
                "total_time": time,
                "gflops": round(gflops, 2),
                "timestamp": datetime.now().isoformat()
            }
        else:
            print(f"--- ERROR: Could not parse output for {version}. ---")
            print("--- Raw Output ---")
            print(output)
            return None

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"--- ERROR: Failed to run {version}. ---")
        # Provide a more helpful error message for FileNotFoundError
        if isinstance(e, FileNotFoundError):
            print(f"File not found: {command[0]}")
        else:
            print(e)
            if hasattr(e, 'stderr'):
                print(e.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Run and collect performance results for matrix multiplication algorithms.")
    parser.add_argument("--versions", nargs="+", default=list(CONFIG.keys()),
                        help=f"Which versions to run. Default is all. Available: {', '.join(CONFIG.keys())}")
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="Number of times to run each version. Default is 1.")
    parser.add_argument("--output_csv", type=str, default="performance_results.csv",
                        help="Path to the output CSV file.")
    
    args = parser.parse_args()

    valid_versions = []
    for v in args.versions:
        if v in CONFIG:
            valid_versions.append(v)
        else:
            print(f"Warning: Version '{v}' is not recognized. Skipping.")

    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, 'a', newline='') as csvfile:
        # Add an 'iteration' column so every single run is logged separately
        fieldnames = ["version", "iteration", "n", "multiplications", "total_time", "gflops", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for version in valid_versions:
            if not compile_code(version):
                continue

            total_multiplications = 0
            total_time_spent = 0
            n_size = None
            successful_runs = 0

            print(f"--- Starting {args.num_iterations} iterations for {version}... ---")
            for i in range(args.num_iterations):
                iteration_num = i + 1
                print(f"--- Running iteration {iteration_num}/{args.num_iterations} for {version}... ---")
                result_data = run_and_parse(version)

                if result_data:
                    # Write this single iteration as its own CSV row
                    row = {
                        "version": version,
                        "iteration": iteration_num,
                        "n": result_data["n"],
                        "multiplications": result_data["multiplications"],
                        "total_time": result_data["total_time"],
                        "gflops": result_data["gflops"],
                        "timestamp": result_data["timestamp"],
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    print(f"--- Logged iteration {iteration_num} for {version}. ---")

                    # Still keep running totals in case aggregate metrics are desired later
                    total_multiplications += result_data["multiplications"]
                    total_time_spent += result_data["total_time"]
                    if n_size is None:
                        n_size = result_data["n"]
                    successful_runs += 1
                else:
                    print(f"--- Iteration {iteration_num} for {version} failed. ---")

            if successful_runs > 1:
                # Optionally print an aggregate summary to console (not written to CSV)
                gflops = (2 * (n_size**3) * total_multiplications) / (total_time_spent * 1e9) if total_time_spent > 0 else 0
                print(f"--- Completed {successful_runs}/{args.num_iterations} successful iterations for {version}. Aggregate GFLOPS: {round(gflops,2)} ---")
            elif successful_runs == 1:
                print(f"--- Completed 1 successful iteration for {version}. ---")
            else:
                print(f"--- All iterations for {version} failed. Nothing to log. ---")

if __name__ == "__main__":
    main()