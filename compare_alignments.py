"""Score two 11-record FASTA files and save one CSV with 20 comparisons.

The first record is the query. The other ten are the candidates.
The first file uses Needleman-Wunsch; the second uses Smith-Waterman.
Pass the same FASTA file twice to compare both methods on identical inputs.
"""
import argparse
import csv
import time
import tracemalloc
from pathlib import Path

from Bio import SeqIO
from needleman_wunsch import needleman_wunsch
from smith_waterman import smith_waterman


def load_fasta(path):
    """Read one query and ten candidates, all containing only A, C, G or T."""
    with Path(path).open(encoding="utf-8") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    if len(records) != 11:
        raise ValueError(f"{path}: expected 11 DNA sequences, found {len(records)}")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: each FASTA record needs a unique ID")

    entries = []
    for record in records:
        dna = str(record.seq).upper()
        if not dna or set(dna) - set("ACGT"):
            raise ValueError(f"{path}: {record.id} must contain only A, C, G, T")
        entries.append((record.id, dna))
    return entries


def measure(query, candidate, algorithm):
    """Time one run and measure peak Python-traced memory in a separate run."""
    # Running memory tracing during the timed measurement makes it much slower.
    start = time.perf_counter_ns()
    score = algorithm(query, candidate)
    time_ms = (time.perf_counter_ns() - start) / 1_000_000

    tracemalloc.start()
    try:
        memory_score = algorithm(query, candidate)
        _, peak_memory = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if score != memory_score:
        raise RuntimeError("Scores differed between timing and memory runs")
    return score, round(time_ms, 6), peak_memory


def run_file(fasta, algorithm, algorithm_name):
    entries = load_fasta(fasta)
    query_id, query = entries[0]
    rows = []
    for candidate_id, candidate in entries[1:]:
        score, time_ms, memory_bytes = measure(query, candidate, algorithm)
        rows.append({
            "algorithm": algorithm_name,
            "query_id": query_id,
            "candidate_id": candidate_id,
            "query_length": len(query),
            "candidate_length": len(candidate),
            "score": score,
            "time_ms": time_ms,
            "peak_python_memory_bytes": memory_bytes,
        })
    # Every candidate is scored before we label the best match.
    best_score = max(row["score"] for row in rows)
    for row in rows:
        row["best_match"] = row["score"] == best_score
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("needleman_fasta", help="11-record FASTA for Needleman-Wunsch")
    parser.add_argument("smith_fasta", help="11-record FASTA for Smith-Waterman")
    parser.add_argument("--output", help="CSV filename; default: output/<first FASTA stem>.csv")
    args = parser.parse_args()

    try:
        rows = (
            run_file(args.needleman_fasta, needleman_wunsch, "Needleman-Wunsch")
            + run_file(args.smith_fasta, smith_waterman, "Smith-Waterman")
        )
        output = (Path(args.output) if args.output else
                  Path("output") / f"{Path(args.needleman_fasta).stem}.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"Saved {len(rows)} comparison rows to {output}")


if __name__ == "__main__":
    main()
