"""Create independent synthetic DNA query replicates as 11-record FASTA files.

The ten baseline candidates are identical to their query, intentionally: mutation.py
introduces random mutations independently in each candidate. Statistical independence
is at the reference-replicate level, not between candidate measurements sharing a query.
"""
import argparse
import hashlib
import random
from pathlib import Path

BASES = 'ACGT'


def create_references(output, lengths, replicates, seed, verbose=False):
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for n in lengths:
        for rep in range(1, replicates + 1):
            # Deterministic independent PRNG stream per length/replicate.
            key = f'{seed}|{n}|{rep}'.encode('utf-8')
            stream_seed = int.from_bytes(hashlib.sha256(key).digest()[:16], 'big')
            rng = random.Random(stream_seed)
            query = ''.join(rng.choice(BASES) for _ in range(n))
            filename = output / f'ref_{n:04d}bp_r{rep:02d}.fasta'
            if filename.exists():
                raise FileExistsError(f'Refusing to overwrite: {filename}')
            with filename.open('w', encoding='utf-8') as f:
                for rec in ['query'] + [f'candidate_{i}' for i in range(1, 11)]:
                    f.write(f'>{rec}\n')
                    for j in range(0, n, 80):
                        f.write(query[j:j+80] + '\n')
            count += 1
            if verbose:
                print(f'Created {filename}')
    print(f'Finished: {count} independently generated query sequences, each with 10 initial copies.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='sequences_independent')
    parser.add_argument('--lengths', type=int, nargs='+', default=[100,250,500])
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--seed', type=int, default=374)
    parser.add_argument('--verbose', action='store_true', help='Print every FASTA filename')
    args = parser.parse_args()
    if any(n < 1 for n in args.lengths) or args.replicates < 1:
        parser.error('Lengths and replicates must be positive integers')
    try:
        create_references(Path(args.out), sorted(set(args.lengths)), args.replicates, args.seed, args.verbose)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
