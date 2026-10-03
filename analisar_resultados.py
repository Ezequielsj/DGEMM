import csv
import statistics
from pathlib import Path


def read_rows(filename):
    with Path(filename).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize(rows):
    grouped = {}
    for row in rows:
        if row.get("validation") not in (None, "", "PASS"):
            continue
        version = row.get("variant") or row.get("versao") or row.get("version")
        key = [version or "versao_desconhecida"]
        if row.get("n"):
            key.append(f"N={row['n']}")
        if row.get("block_size") not in (None, "", "0"):
            key.append(f"bloco={row['block_size']}")
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
        print(f"  min={min(values):.2f}; max={max(values):.2f} GFLOPS")


if __name__ == "__main__":
    for part in (1, 2, 3):
        filename = f"resultados_parte{part}.csv"
        if Path(filename).exists():
            rows = read_rows(filename)
            print(f"Parte {part}: {len(rows)} registros")
            summarize(rows)
            print()

    campaign_file = Path("resultados_campanha.csv")
    if campaign_file.exists():
        rows = read_rows(campaign_file)
        print(f"Campanha ampliada: {len(rows)} registros")
        summarize(rows)
