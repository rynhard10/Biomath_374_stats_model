"""Create 11-record FASTA mutation datasets.

One FASTA contains 1 unchanged query + 10 candidates.

Usage:
    python generate_mutations.py sequences --type substitution
    python generate_mutations.py sequences --type all --compare
    python generate_mutations.py sequences --type deletion --pattern 'sequences_0100bp.fasta' --rates 0 0.05 --lengths 1 5 --compare

Without --compare, generates FASTA and metadata ONLY. With --compare, runs the
existing compare_alignments.py on each FASTA and saves one CSV per FASTA in
results/{original,substitution,deletion,insertion}/.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

BASES = "ACGT"
DEFAULT_RATES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
DEFAULT_LENGTHS = (1, 5, 10, 20)
TYPES = ("substitution", "deletion", "insertion")
MANIFEST_FIELDS = (
    "source_fasta", "generated_fasta", "query_id", "candidate_id",
    "mutation_type", "mutation_rate", "mutation_block_length",
    "initial_length", "final_length", "mutated_bases", "mutation_blocks",
    "seed", "events_json",
)
RESULT_FIELDS = (
    "source_fasta", "mutation_type", "mutation_rate",
    "mutation_block_length", "mutated_bases", "mutation_blocks", "seed",
)


def read_fasta(path):
    with path.open(encoding="utf-8") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    if len(records) != 11:
        raise ValueError(f"{path}: expected 11 FASTA records; got {len(records)}")
    if len({r.id for r in records}) != len(records):
        raise ValueError(f"{path}: FASTA record IDs must be unique")
    for record in records:
        sequence = str(record.seq).upper()
        if not sequence or any(c not in BASES for c in sequence):
            raise ValueError(f"{path}: invalid DNA sequence for {record.id}; use only ACGT")
    return records


def binomial_bases(n, rate, rng):
    """Draw number of affected bases from Binomial(n, rate), stdlib only."""
    return sum(rng.random() < rate for _ in range(n))


def blocks_for_count(k, block_length, rng):
    """Partition K affected bases into blocks, usually of length L."""
    blocks = [block_length] * (k // block_length)
    if k % block_length:
        blocks.append(k % block_length)
    rng.shuffle(blocks)
    return blocks


def disjoint_starts(n, blocks, rng):
    """Choose disjoint block positions in the original candidate sequence.

    Prefer at least one untouched base between blocks when there is room.
    Coordinates are 0-based, in the ORIGINAL (pre-mutation) candidate.
    """
    if not blocks:
        return []
    k = sum(blocks)
    b = len(blocks)
    if k > n:
        raise ValueError("Cannot change/delete more bases than the sequence contains")
    minimum_gap = 1 if k + b - 1 <= n else 0
    extras = n - k - minimum_gap * (b - 1)

    # Uniform stars-and-bars distribution of remaining free positions across b+1 gaps.
    separators = sorted(rng.sample(range(extras + b), b))
    gaps = []
    previous = -1
    for separator in separators:
        gaps.append(separator - previous - 1)
        previous = separator
    gaps.append((extras + b) - previous - 1)

    starts = []
    pos = gaps[0]
    for i, block_size in enumerate(blocks):
        starts.append(pos)
        pos += block_size
        if i < b - 1:
            pos += minimum_gap + gaps[i + 1]
    return starts


def mutate_sequence(sequence, mutation_type, rate, block_length, rng):
    """Return (mutated_sequence, number_of_bases_affected, event_list)."""
    n = len(sequence)
    k = binomial_bases(n, rate, rng)
    blocks = blocks_for_count(k, block_length, rng)
    if not blocks:
        return sequence, 0, []

    events = []
    if mutation_type in ("substitution", "deletion"):
        starts = disjoint_starts(n, blocks, rng)
        intervals = sorted(zip(starts, blocks))
        parts = []
        previous_end = 0
        for start, length in intervals:
            parts.append(sequence[previous_end:start])
            original = sequence[start:start + length]
            if mutation_type == "substitution":
                replacement = "".join(
                    rng.choice([b for b in BASES if b != old]) for old in original
                )
                parts.append(replacement)
                events.append({"start": start, "length": length,
                               "original": original, "replacement": replacement})
            else:
                events.append({"start": start, "length": length, "deleted": original})
            previous_end = start + length
        parts.append(sequence[previous_end:])
        return "".join(parts), k, events

    if mutation_type == "insertion":
        # Insertions occur BETWEEN original nucleotides, including both ends.
        sites = rng.sample(range(n + 1), len(blocks))
        inserted_at = {}
        for site, length in zip(sites, blocks):
            added = "".join(rng.choice(BASES) for _ in range(length))
            inserted_at[site] = added
            events.append({"site": site, "length": length, "inserted": added})
        parts = []
        for site in range(n + 1):
            if site in inserted_at:
                parts.append(inserted_at[site])
            if site < n:
                parts.append(sequence[site])
        events.sort(key=lambda e: e["site"])
        return "".join(parts), k, events

    raise ValueError(f"Unknown mutation type: {mutation_type}")


def stable_rng(seed, source_name, candidate_id, mutation_type, rate, length):
    """Reproducible per-candidate seed, stable across processes and run order."""
    key = f"{seed}|{source_name}|{candidate_id}|{mutation_type}|{rate:.12g}|{length}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def rate_tag(rate):
    return "p" + f"{rate * 100:g}".replace(".", "p") + "pct"


def record_copy(record, seq):
    return SeqRecord(Seq(seq), id=record.id, description=record.description)


def manifest_row(source, generated, query_id, record_id, mutation_type, rate,
                 block_length, old_len, new_len, k, events, seed):
    return {
        "source_fasta": source.name,
        "generated_fasta": str(generated),
        "query_id": query_id,
        "candidate_id": record_id,
        "mutation_type": mutation_type,
        "mutation_rate": rate,
        "mutation_block_length": block_length,
        "initial_length": old_len,
        "final_length": new_len,
        "mutated_bases": k,
        "mutation_blocks": len(events),
        "seed": seed,
        "events_json": json.dumps(events, separators=(",", ":")),
    }


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def augment_result_csv(csv_path, metadata_by_candidate):
    """Add experimental conditions to the existing compare program's CSV.

    Does not change NW/SW algorithms or their time/memory measurements.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        data = list(reader)
    if len(data) != 20:
        raise ValueError(f"{csv_path}: expected 20 comparisons; found {len(data)}")
    for row in data:
        meta = metadata_by_candidate[row["candidate_id"]]
        row.update({field: meta[field] for field in RESULT_FIELDS})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=original_fields + list(RESULT_FIELDS))
        writer.writeheader()
        writer.writerows(data)


def run_comparison(fasta_file, category, results_dir, compare_script, meta):
    """Run the same FASTA through both algorithms and save the measurements."""
    compare_script = compare_script.resolve()
    if not compare_script.is_file():
        raise FileNotFoundError(f"Missing comparison program: {compare_script}")
    output_csv = results_dir / category / (fasta_file.stem + ".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(compare_script), str(fasta_file.resolve()),
               str(fasta_file.resolve()), "--output", str(output_csv.resolve())]
    finished = subprocess.run(command, cwd=compare_script.parent,
                              capture_output=True, text=True)
    if finished.returncode != 0:
        raise ValueError(f"Comparison failed for {fasta_file.name}: "
                         f"{finished.stderr or finished.stdout}")
    augment_result_csv(output_csv, meta)


def select_input_files(folder, pattern):
    if pattern != "all":
        paths = sorted(p for p in folder.glob(pattern) if p.is_file())
    else:
        paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".fasta", ".fa", ".fna"}
        )
    if not paths:
        raise ValueError(f"No FASTA files found in {folder} (pattern={pattern})")
    return paths


def generate(args):
    input_folder = Path(args.input_folder).resolve()
    if not input_folder.is_dir():
        raise ValueError(f"Input folder not found: {input_folder}")
    generated_dir = Path(args.generated_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    source_files = select_input_files(input_folder, args.pattern)
    if len({p.stem for p in source_files}) != len(source_files):
        raise ValueError("FASTA filenames must have distinct stems to prevent overwrites")
    mutation_types = TYPES if args.type == "all" else (args.type,)
    rates = sorted(set(args.rates))
    lengths = sorted(set(args.lengths))
    if any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError("Mutation rates must be between 0 and 1")
    if any(length < 1 for length in lengths):
        raise ValueError("Mutation block lengths must be positive integers")

    original_rows = []
    mutation_rows = {name: [] for name in mutation_types}
    generated_count = 0
    compared_count = 0

    for source in source_files:
        records = read_fasta(source)
        query_id = records[0].id
        baseline = generated_dir / "original" / source.name
        baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, baseline)
        original_meta = {}
        for candidate in records[1:]:
            original_meta[candidate.id] = manifest_row(
                source, baseline, query_id, candidate.id, "original", 0, 0,
                len(candidate.seq), len(candidate.seq), 0, [], args.seed
            )
        original_rows.extend(original_meta.values())
        generated_count += 1
        if args.compare:
            run_comparison(baseline, "original", results_dir, Path(args.compare_script), original_meta)
            compared_count += 1

        for mutation_type in mutation_types:
            for rate in rates:
                for length in lengths:
                    stem = f"{source.stem}_{mutation_type}_{rate_tag(rate)}_L{length:02d}"
                    output = generated_dir / mutation_type / f"{stem}.fasta"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    new_records = [record_copy(records[0], str(records[0].seq).upper())]
                    meta = {}
                    for record in records[1:]:
                        old = str(record.seq).upper()
                        rng = stable_rng(args.seed, source.name, record.id, mutation_type, rate, length)
                        new, k, events = mutate_sequence(old, mutation_type, rate, length, rng)
                        if not new:
                            raise ValueError(f"Mutation produced empty sequence: {source.name}/{record.id}")
                        new_records.append(record_copy(record, new))
                        meta[record.id] = manifest_row(
                            source, output, query_id, record.id, mutation_type,
                            rate, length, len(old), len(new), k, events, args.seed
                        )
                    with output.open("w", encoding="utf-8") as handle:
                        SeqIO.write(new_records, handle, "fasta")
                    mutation_rows[mutation_type].extend(meta.values())
                    generated_count += 1
                    if args.compare:
                        run_comparison(output, mutation_type, results_dir, Path(args.compare_script), meta)
                        compared_count += 1
    
    write_manifest(metadata_dir / "original.csv", original_rows)
    for kind, rows in mutation_rows.items():
        write_manifest(metadata_dir / f"{kind}.csv", rows)

    print(f"Finished: {generated_count} FASTA files from {len(source_files)} input file(s).")
    print(f"Mutated files: {len(source_files)*len(rates)*len(lengths)} per requested mutation type.")
    print(f"Generated FASTA: {generated_dir}; metadata: {metadata_dir}")
    if args.compare:
        print(f"Generated {compared_count} comparison CSV files in {results_dir}.")
    else:
        print("No CSV measurements yet: rerun with --compare, or run compare_alignments.py manually.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate mutation FASTA datasets; optionally run NW and SW benchmarks."
    )
    parser.add_argument("input_folder", help="Folder of original 11-record FASTA files (not CSVs)")
    parser.add_argument("--type", choices=(*TYPES, "all"), required=True,
                        help="Mutation type to generate; all generates all three")
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES,
                        help="Base-level mutation fractions as probabilities (default: 0 .01 .02 .05 .10 .20)")
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
                        help="Contiguous mutation block sizes in bp (default: 1 5 10 20)")
    parser.add_argument("--seed", type=int, default=374, help="Deterministic simulation seed")
    parser.add_argument("--pattern", default="all", help="Input FASTA filename glob; default: all FASTA extensions")
    parser.add_argument("--generated-dir", default="generated/fastas", help="Output root for FASTA files")
    parser.add_argument("--results-dir", default="generated/results", help="Output root for measured CSV files")
    parser.add_argument("--metadata-dir", default="generated/metadata", help="Output root for mutation event metadata")
    parser.add_argument("--compare", action="store_true", help="Also run compare_alignments.py to create measured CSV files")
    parser.add_argument("--compare-script", default="compare_alignments.py",
                        help="Path to existing alignment comparison script")
    args = parser.parse_args()
    try:
        generate(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
