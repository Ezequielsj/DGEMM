import csv
import statistics
from pathlib import Path


def read_rows(filename):
    with Path(filename).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize(rows):
    grouped = {}
    for row in rows:
        key = [row["versao"]]
        if row.get("n"):
            key.append(f"N={row['n']}")
        if row.get("threads"):
            key.append(f"threads={row['threads']}")
        grouped.setdefault("; ".join(key), []).append(float(row["gflops"]))

    for version, values in grouped.items():
        mean = statistics.mean(values)
        median = statistics.median(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"{version}: mean={mean:.2f} GFLOPS; "
            f"median={median:.2f}; std={deviation:.2f}; samples={len(values)}"
        )


if __name__ == "__main__":
    for part in (1, 2, 3):
        rows = read_rows(f"resultados_parte{part}.csv")
        print(f"Parte {part}: {len(rows)} registros")
        summarize(rows)
        print()
