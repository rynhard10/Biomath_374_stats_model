# Biomaths 374 — DNA mutations and alignment efficiency

The algorithms use full dynamic-programming matrices. They return scores only, not the aligned strings: the score changes are not a measure of nucleotide-level alignment accuracy.

## Files and folders

| Path | What it does |
|---|---|
| `needleman_wunsch.py` | Global alignment score, with the +2/−1/−2 scoring rules |
| `smith_waterman.py` | Local alignment score, using the same rules |
| `compare_alignments.py` | Reads two 11-record FASTA files, tests 10 candidates per algorithm and saves 20 measurement rows |
| `generate_references.py` | Generates independent random reference sequences (one query plus ten initial identical candidate copies) |
| `generate_mutations.py` | Applies independently sampled mutation burdens to each candidate; creates mutated FASTAs, metadata and optionally runs the alignment comparisons |
| `analyze_mutations.py` | Checks the CSVs, removes repeated zero-mutation treatments, produces graphs and fits exploratory regressions |
| `data/references/` | 15 original FASTA files used in the reported experiment |
| `data/results/` | 1,095 original measurement CSV files; preserved without changes |
| `analysis/` | The saved completeness, summary and regression outputs from the reported experiment |
| `tests/` | Checks for algorithm scores, reproducible mutations, FASTA input and CSV output |

Each reference FASTA has 11 records: the first record (`query`) is the fixed reference; the other 10 are candidates. The mutation generator changes only the candidates, never the query.

## 1. Install (Windows, macOS or Linux)

The code is cross-platform: it uses Python rather than Linux-only system commands. Python 3.10 or newer is recommended. A virtual environment is recommended but must not be committed to GitHub.
No network access is required after dependencies are installed.

### Windows (PowerShell)

Open PowerShell in the extracted repository folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If `py` is unavailable, install Python from python.org and use `python` instead. You do **not** need to activate the environment when you call its Python executable directly.

### macOS and Linux (Terminal)

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m unittest discover -s tests -v
```

On some Linux distributions, the Python `venv` module must first be installed through the system package manager. On Arch Linux, you can use the commands above directly if Python is installed.

For the commands below, replace `python` with `./.venv/bin/python` (macOS/Linux) or `.\.venv\Scripts\python.exe` (Windows PowerShell) if you have not activated the environment. Alternatively, activate it (`source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`) and use `python`.

## 2. Run one alignment comparison (quick check)

The same FASTA file is passed to both algorithms so they receive identical input sequences:

```bash
python compare_alignments.py data/references/ref_0050bp_r01.fasta data/references/ref_0050bp_r01.fasta --output output/example.csv
```

Expected: one CSV at `output/example.csv` containing 20 result rows (plus the header). On the first run, the program prints only the output filename and record count.

Columns: `algorithm`, `query_id`, `candidate_id`, `query_length`, `candidate_length`, `score`, `time_ms`, `peak_python_memory_bytes`, `best_match`.

`time_ms` is one timed run using `time.perf_counter_ns()`. Memory is measured in a second Runtime using `tracemalloc`, which tracks Python allocations and is not total system RAM. FASTA loading and CSV writing are excluded from the timed part. Each candidate is scored, even if an earlier candidate already has a high score.

## 3. Test mutation generation (small, quick check)

This creates only a few mutated datasets from one 50-bp reference. Nothing under `data/` is changed:

```bash
python generate_mutations.py data/references --pattern ref_0050bp_r01.fasta --type all --rates 0 0.05 --lengths 1 --seed 2026 --compare --generated-dir generated/fastas --results-dir generated/results --metadata-dir generated/metadata
```

Look in `generated/fastas/{original,substitution,deletion,insertion}/`, `generated/results/` and `generated/metadata/`. Every result CSV has 20 alignment rows. These generated files are intentionally excluded from Git because the input references and random seeds reproduce them.

The mutation count for each candidate is sampled independently as K ~ Binomial(n, p). The `K` affected bases are grouped into contiguous blocks of up to the chosen `--lengths` size. Thus `p` is the affected-base probability, not the chance that a new mutation block begins. At positive `p`, zero mutations can still occur.

## 4. Reproduce the full simulation (slow)

```bash
python generate_references.py --out generated/references --lengths 50 100 250 500 1000 --replicates 3 --seed 374
```

Then reproduce all mutations and performance measurements:

```bash
python generate_mutations.py generated/references --type all --rates 0 0.01 0.02 0.05 0.10 0.20 --lengths 1 5 10 20 --seed 2026 --compare --generated-dir generated/fastas --results-dir generated/results --metadata-dir generated/metadata
```

This makes **1,095 FASTA/result datasets**, and the comparisons involve many full dynamic-programming matrices. It can take hours, especially for 1,000-bp sequences. Runtime and peak allocations on another computer will not necessarily match the submitted CSVs.

## 5. Recreate the analysis and graphs

```bash
python analyze_mutations.py --results data/results --max-length 1000 --expected-rates 0 0.01 0.02 0.05 0.10 0.20 --expected-lengths 1 5 10 20 --balanced-only --out analysis_outputs
```

This checks that all conditions are present, removes duplicated zero-rate treatment datasets and produces `master_data.csv`, `descriptive_summary.csv`, `model_coefficients.csv` and figures in `analysis_outputs/`. The existing `data/results/` and saved report files are not modified

```bash
python analyze_mutations.py --results data/results --max-length 1000 --expected-rates 0 0.01 0.02 0.05 0.10 0.20 --expected-lengths 1 5 10 20 --balanced-only --cluster-by reference --out analysis_outputs_reference_clusters
```

The saved `analysis/` files come from the original pair-cluster fit.

