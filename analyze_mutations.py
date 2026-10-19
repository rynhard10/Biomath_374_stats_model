#!/usr/bin/env python3
"""Analyse mutation measurements from the 11-sequence FASTA experiment.

Checks completeness, removes duplicate zero-mutation conditions, generates
summary graphs and fits exploratory regression models. The original CSVs are
never changed. The random-effect model can fail; this is reported, not hidden.

    python analyze_mutations.py --results data/results --balanced-only
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

TYPES = ('substitution', 'deletion', 'insertion')
RATES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
BLOCK_LENGTHS = (1, 5, 10, 20)
ALGOS = ('Needleman-Wunsch', 'Smith-Waterman')
REQUIRED = {
    'algorithm', 'query_id', 'candidate_id', 'query_length', 'candidate_length',
    'score', 'time_ms', 'peak_python_memory_bytes', 'mutation_type',
    'mutation_rate', 'mutation_block_length', 'mutated_bases', 'source_fasta'
}
NUMERIC = ('query_length', 'candidate_length', 'score', 'time_ms',
           'peak_python_memory_bytes', 'mutation_rate',
           'mutation_block_length', 'mutated_bases')


def inspect_file(path: Path):
    """Return a fully validated 20-row dataframe or a failure reason."""
    try:
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            return None, f'missing fields: {", ".join(sorted(missing))}'
        if len(frame) != 20:
            return None, f'expected 20 rows, found {len(frame)}'
        for name in NUMERIC:
            frame[name] = pd.to_numeric(frame[name], errors='raise')
        if frame[list(NUMERIC)].isna().any().any():
            return None, 'missing numeric measurements'
        if set(frame['algorithm']) != set(ALGOS):
            return None, 'incorrect algorithm names'
        if any((frame['algorithm'] == name).sum() != 10 for name in ALGOS):
            return None, 'expected 10 candidates for each algorithm'
        if frame.duplicated(['algorithm', 'candidate_id']).any():
            return None, 'repeated candidate within one algorithm'
        a = set(frame.loc[frame['algorithm'] == ALGOS[0], 'candidate_id'])
        b = set(frame.loc[frame['algorithm'] == ALGOS[1], 'candidate_id'])
        if a != b or len(a) != 10:
            return None, 'algorithms do not have the same 10 candidates'
        for name in ('query_length', 'source_fasta', 'mutation_type',
                     'mutation_rate', 'mutation_block_length'):
            if frame[name].nunique() != 1:
                return None, f'inconsistent {name} within file'
        kind = frame['mutation_type'].iat[0]
        if kind not in (*TYPES, 'original'):
            return None, f'unsupported mutation_type={kind}'
        if (frame['query_length'] <= 0).any() or (frame['candidate_length'] <= 0).any():
            return None, 'nonpositive sequence length'
        if (frame['time_ms'] <= 0).any() or (frame['peak_python_memory_bytes'] <= 0).any():
            return None, 'nonpositive runtime/memory'
        if not frame['mutation_rate'].between(0, 1).all():
            return None, 'mutation rate outside [0,1]'
        if (frame['mutated_bases'] < 0).any():
            return None, 'negative mutation count'
        frame['result_file'] = str(path)
        return frame, 'OK'
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        return None, f'unreadable CSV: {str(error)[:180]}'


def load_results(folder: Path, max_length: int):
    paths = sorted(folder.rglob('*.csv'))
    if not paths:
        raise ValueError(f'No CSV files found under {folder}')
    good, audit = [], []
    for path in paths:
        df, why = inspect_file(path)
        if df is None:
            audit.append({'file': str(path), 'status': 'SKIPPED', 'details': why})
        elif df['query_length'].iat[0] > max_length:
            audit.append({'file': str(path), 'status': 'EXCLUDED',
                          'details': f'query length > {max_length}'})
        else:
            good.append(df)
            audit.append({'file': str(path), 'status': 'INCLUDED', 'details': '20 valid rows'})
    if not good:
        raise ValueError('No valid result CSVs within the requested length cutoff. '
                         'Inspect validation_log.csv after the script is rerun.')
    return pd.concat(good, ignore_index=True), pd.DataFrame(audit)


def check_completeness(df, rates, block_lengths):
    # Check all expected files for every reference source represented in valid data.
    sources = sorted(df['source_fasta'].unique())
    actual = set(tuple(x) for x in df[
        ['source_fasta', 'mutation_type', 'mutation_rate', 'mutation_block_length']
    ].drop_duplicates().itertuples(index=False, name=None))
    checks = []
    for source in sources:
        candidates = [('original', 0., 0)]
        candidates.extend(itertools.product(TYPES, rates, block_lengths))
        for kind, rate, length in candidates:
            checks.append({
                'source_fasta': source, 'mutation_type': kind,
                'mutation_rate': rate, 'mutation_block_length': length,
                'status': 'complete' if (source, kind, rate, length) in actual else 'missing'
            })
    return pd.DataFrame(checks)


def remove_duplicate_baselines(df):
    """At p=0, all mutation types and block lengths generate unchanged inputs.
    Keep original if present. If missing, keep just one p=0 dataset for that source.
    Intention-to-treat p>0 rows with K=0 are kept, since they are valid draws.
    """
    identity = ['source_fasta', 'algorithm', 'query_id', 'candidate_id']
    df = df.copy()
    df['is_zero_condition'] = df['mutation_rate'].eq(0)
    baseline_candidates = df.loc[df['is_zero_condition']].copy()
    baseline_candidates['pref'] = np.where(
        baseline_candidates['mutation_type'].eq('original'), 0, 1)
    baseline_candidates = baseline_candidates.sort_values(
        ['pref', 'mutation_type', 'mutation_block_length', 'result_file'])
    baseline = baseline_candidates.drop_duplicates(identity, keep='first').drop(
        columns=['pref'])
    nonzero = df.loc[~df['is_zero_condition']]
    clean = pd.concat([baseline, nonzero], ignore_index=True)
    # Refuse silent duplicate output from reruns/copying files.
    condition = identity + ['mutation_type', 'mutation_rate', 'mutation_block_length']
    repeats = clean.duplicated(condition, keep=False)
    if repeats.any():
        raise ValueError(f'{int(repeats.sum())} duplicated rows after removing p=0 replicates. '
                         'Check for duplicate result CSVs in results/.')
    return clean, baseline


def add_variables(df, baseline):
    df = df.copy()
    index = ['source_fasta', 'algorithm', 'query_id', 'candidate_id']
    b = baseline[index + ['score', 'candidate_length']].copy().rename(columns={
        'score': 'original_score', 'candidate_length': 'initial_length'
    })
    df = df.merge(b, on=index, how='left', validate='many_to_one')
    df['group_id'] = df['source_fasta'].astype(str) + '::' + df['query_id'].astype(str) + '::' + df['candidate_id'].astype(str)
    df['matrix_cells'] = (df['query_length'] + 1) * (df['candidate_length'] + 1)
    df['log_cells'] = np.log(df['matrix_cells'])
    df['log_time'] = np.log(df['time_ms'])
    df['memory_mib'] = df['peak_python_memory_bytes'] / (1024**2)
    df['log_memory'] = np.log(df['memory_mib'])
    # If the original baseline is missing, the mutation program still allows us
    # to reconstruct original candidate length from its known indel count.
    inferred = np.where(df['mutation_type'].eq('insertion'),
                        df['candidate_length'] - df['mutated_bases'],
                        np.where(df['mutation_type'].eq('deletion'),
                                 df['candidate_length'] + df['mutated_bases'],
                                 df['candidate_length']))
    df['initial_length'] = df['initial_length'].fillna(pd.Series(inferred, index=df.index))
    df['realized_rate'] = df['mutated_bases'] / df['initial_length']
    df['log_block_length'] = np.log(np.maximum(1, df['mutation_block_length']))
    df['block_effect'] = df['realized_rate'] * df['log_block_length']
    df['p_block_effect'] = df['mutation_rate'] * df['log_block_length']
    df['sw'] = df['algorithm'].eq('Smith-Waterman').astype(int)
    for kind, prefix in (('substitution', 'sub'), ('deletion', 'del'), ('insertion', 'ins')):
        df[f'{prefix}_burden'] = np.where(df['mutation_type'].eq(kind), df['realized_rate'], 0.)
        df[f'{prefix}_p'] = np.where(df['mutation_type'].eq(kind), df['mutation_rate'], 0.)
    df['score_degradation'] = (df['original_score'] - df['score']) / (2 * df['query_length'])
    return df


def summary_table(df):
    keys = ['mutation_type', 'mutation_rate', 'mutation_block_length', 'query_length', 'algorithm']
    summary = df.groupby(keys, dropna=False).agg(
        comparisons=('candidate_id', 'size'),
        median_time_ms=('time_ms', 'median'),
        mean_time_ms=('time_ms', 'mean'),
        median_memory_mib=('memory_mib', 'median'),
        mean_score=('score', 'mean'),
        mean_score_degradation=('score_degradation', 'mean'),
        mean_realized_rate=('realized_rate', 'mean'),
        mean_matrix_cells=('matrix_cells', 'mean')
    ).reset_index()
    return summary


def fits_file(out, name, formula, df, model_text, coeff_rows, diagnostics, cluster_by="pair"):
    data = df.dropna(subset=[x for x in ['log_time', 'log_memory'] if x in formula] +
                     ['log_cells', 'realized_rate', 'group_id']).copy()
    # Score model additionally needs a matched baseline.
    if name == 'score':
        data = data.dropna(subset=['score_degradation'])
    if data.empty or data['group_id'].nunique() < 3:
        model_text.append(f'\n{name}: INSUFFICIENT data/groups to fit.\n')
        return
    # Center logcells for numerical conditioning; slopes remain unchanged.
    data['centered_log_cells'] = data['log_cells'] - data['log_cells'].mean()
    actual_formula = formula.replace('log_cells', 'centered_log_cells')
    # A singular fixed-effect design cannot support independent coefficients.
    probe = smf.ols(actual_formula, data)
    design = probe.exog
    rank = np.linalg.matrix_rank(design)
    if rank < design.shape[1]:
        model_text.append(
            f'\n{name}: MODEL NOT IDENTIFIABLE: design rank {rank}/{design.shape[1]}.\n'
            'Use several completed sequence lengths or simplify the predictors.\n'
        )
        return
    cond = np.linalg.cond(design)
    chosen = 'random intercept by original source/candidate pair'
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Random effects covariance is singular')
            warnings.filterwarnings('error', category=ConvergenceWarning)
            fitted = smf.mixedlm(actual_formula, data, groups=data['group_id']).fit(
                reml=False, method='lbfgs', maxiter=250, disp=False)
        if not fitted.converged or not np.isfinite(fitted.fe_params).all() or float(fitted.cov_re.iloc[0, 0]) < 1e-9:
            raise ValueError('mixed model did not converge cleanly')
        coefs = fitted.fe_params
        ses = fitted.bse_fe
        pvals = fitted.pvalues.loc[coefs.index]
        conf = fitted.conf_int().loc[coefs.index]
        fitted_y = fitted.fittedvalues
        residuals = data[formula.split('~')[0].strip()] - fitted_y
        model_text.append(f'\n===== {name.upper()} =====\n{chosen}\n'
                          f'Groups: {data["group_id"].nunique()}, rows: {len(data)}\n'
                          f'Formula: {actual_formula}\nDesign condition number: {cond:.4g}\n{fitted.summary()}\n')
    except Exception as error:
        cluster = data["source_fasta"] if cluster_by == "reference" else data["group_id"]
        chosen = f"FALLBACK: OLS with clustered standard errors by {cluster_by}"
        fitted = smf.ols(actual_formula, data).fit(
            cov_type="cluster", cov_kwds={"groups": cluster})
        coefs = fitted.params
        ses = fitted.bse
        pvals = fitted.pvalues
        conf = fitted.conf_int()
        fitted_y = fitted.predict(data)
        residuals = data[formula.split('~')[0].strip()] - fitted_y
        model_text.append(f'\n===== {name.upper()} =====\n{chosen}\n'
                          f'LMM unavailable: {error}\n'
                          f'Groups: {data["group_id"].nunique()}, rows: {len(data)}\n'
                          f'Formula: {actual_formula}\nDesign condition number: {cond:.4g}\n{fitted.summary()}\n')
    for term, value in coefs.items():
        coeff_rows.append({
            'model': name, 'fitting_method': chosen, 'term': term,
            'estimate': float(value), 'std_error': float(ses.loc[term]),
            'ci_low_95': float(conf.loc[term].iloc[0]),
            'ci_high_95': float(conf.loc[term].iloc[1]),
            'p_value': float(pvals.loc[term]), 'n_rows': len(data),
            'n_groups': (data['source_fasta'].nunique() if 'by reference' in chosen else data['group_id'].nunique())
        })
    diagnostics[name] = (np.asarray(fitted_y), np.asarray(residuals))


def fit_models(df, out, cluster_by="pair"):
    models = []
    coefficients = []
    diagnostics = {}
    # Runtime: test mutation effect beyond the size of the dynamic-programming matrix.
    fit_specs = [
        ('runtime', 'log_time ~ log_cells + sw + sub_p + ins_p + del_p + p_block_effect'),
        ('memory', 'log_memory ~ log_cells + sw + sub_p + ins_p + del_p + p_block_effect'),
        ('score', 'score_degradation ~ sw + sub_p + ins_p + del_p + p_block_effect'),
    ]
    for name, formula in fit_specs:
        if name == 'score':
            # The old GAP=0 experiment produced identical scores for NW and SW.
            # Under GAP=-2 this may no longer hold. Keep both algorithms only
            # when at least one matched pair has different scoring outcomes.
            pivot_keys = ['source_fasta','query_id','candidate_id',
                          'mutation_type','mutation_rate','mutation_block_length']
            wide = df.pivot(index=pivot_keys, columns='algorithm', values='score')
            identical = bool(np.allclose(wide[ALGOS[0]], wide[ALGOS[1]], atol=0, rtol=0))
            if identical:
                score_data = df.loc[df['algorithm'].eq(ALGOS[0])].copy()
                formula = ('score_degradation ~ sub_p + ins_p + del_p '
                           '+ p_block_effect')
                models.append('Score outcomes are identical across algorithms; fitting unique NW observations only.\n')
            else:
                score_data = df
                models.append('Score outcomes differ between algorithms; fitting both with algorithm indicator.\n')
            fits_file(out, name, formula, score_data, models, coefficients, diagnostics, cluster_by)
        else:
            fits_file(out, name, formula, df, models, coefficients, diagnostics, cluster_by)
    pd.DataFrame(coefficients, columns=['model','fitting_method','term','estimate','std_error',
                                         'ci_low_95','ci_high_95','p_value','n_rows','n_groups']).to_csv(
        out / 'model_coefficients.csv', index=False)
    for name, (fitted, resid) in diagnostics.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(fitted, resid, alpha=.25, s=8)
        ax.axhline(0, color='black', lw=1, linestyle='--')
        ax.set_xlabel('Fitted response')
        ax.set_ylabel('Residual')
        ax.set_title(f'{name.capitalize()} model: residuals vs fitted values')
        fig.tight_layout()
        fig.savefig(out / f'model_diagnostics_{name}.png', dpi=160)
        plt.close(fig)
    return len(coefficients)


def make_graphs(df, out, plot_length=None):
    if plot_length is None:
        plot_length = int(df['query_length'].max())
    data = df.loc[df['query_length'].eq(plot_length) &
                  df['mutation_block_length'].eq(1)]
    # One graph for each mutation type and outcome; line styles show overlapping values.
    for kind in TYPES:
        selected = data.loc[(data['mutation_type'].eq(kind)) |
                            (data['mutation_type'].eq('original'))].copy()
        if selected.empty:
            continue
        for metric, ylabel, stem in (
            ('time_ms', 'Median execution time (ms)', 'runtime'),
            ('memory_mib', 'Median peak traced memory (MiB)', 'memory'),
            ('score_degradation', 'Mean normalized score degradation', 'score_degradation')
        ):
            summary = selected.groupby(['algorithm', 'mutation_rate'], as_index=False)[metric].agg(
                'mean' if metric == 'score_degradation' else 'median')
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for algo, style in ((ALGOS[0], ('-', 'o')), (ALGOS[1], ('--', 'x'))):
                sub = summary.loc[summary['algorithm'].eq(algo)].sort_values('mutation_rate')
                if not sub.empty:
                    ax.plot(sub['mutation_rate'] * 100, sub[metric],
                            style[0], marker=style[1], linewidth=2, markersize=6, label=algo)
            ax.set(xlabel='Requested mutation probability (%)', ylabel=ylabel,
                   title=f'{kind.capitalize()}, L=1 bp, query length={plot_length} bp')
            ax.grid(alpha=.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f'{stem}_vs_rate_{kind}.png', dpi=160)
            plt.close(fig)
    # Cell-count scaling by algorithm, across all tested sequence lengths/conditions.
    for metric, label, stem in (
        ('time_ms', 'Median execution time (ms)', 'runtime_vs_cells'),
        ('memory_mib', 'Median peak traced memory (MiB)', 'memory_vs_cells')
    ):
        frame = df.groupby(['algorithm', 'query_length', 'mutation_type',
                            'mutation_rate', 'mutation_block_length'], as_index=False).agg(
            cells=('matrix_cells', 'median'), value=(metric, 'median'))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for algo, marker in ((ALGOS[0], 'o'), (ALGOS[1], 'x')):
            sub = frame.loc[frame['algorithm'].eq(algo)]
            ax.scatter(sub['cells'], sub['value'], marker=marker,
                       alpha=.25, s=16, label=algo)
        ax.set(xlabel='Dynamic-programming matrix cells: (n+1)(m+1)',
               ylabel=label, title=f'{label} versus matrix size (all included data)')
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{stem}.png', dpi=160)
        plt.close(fig)
    return plot_length


def run(args):
    folder = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, audit = load_results(folder, args.max_length)
    audit.to_csv(out / 'validation_log.csv', index=False)
    completeness = check_completeness(df, args.expected_rates, args.expected_lengths)
    completeness.to_csv(out / 'completeness.csv', index=False)
    retained_sources = None
    if args.balanced_only:
        complete_counts = completeness.assign(is_complete=completeness['status'].eq('complete')).groupby('source_fasta')['is_complete'].agg(['sum', 'count'])
        retained_sources = list(complete_counts.index[complete_counts['sum'].eq(complete_counts['count'])])
        if not retained_sources:
            raise ValueError('No complete reference datasets for the specified expected rates and block lengths. Check completeness.csv.')
        df = df.loc[df['source_fasta'].isin(retained_sources)].copy()
    clean, baseline = remove_duplicate_baselines(df)
    clean = add_variables(clean, baseline)
    if clean['initial_length'].isna().any():
        n = int(clean['initial_length'].isna().sum())
        print(f'WARNING: {n} rows have no original or p=0 reference and cannot be used for score degradation.')
    clean.to_csv(out / 'master_data.csv', index=False)
    summary_table(clean).to_csv(out / 'descriptive_summary.csv', index=False)
    plotted = make_graphs(clean, out, args.plot_length)
    ncoef = 0 if args.no_models else fit_models(clean, out, args.cluster_by)
    total_csv = len(audit)
    complete = int((completeness['status'] == 'complete').sum())
    expected = len(completeness)
    report = [
        'BIOMATHS 374 MUTATION ANALYSIS — DATA AUDIT',
        '================================================',
        f'Results directory: {folder.resolve()}',
        f'Maximum included query length: {args.max_length} bp',
        f'CSV files found: {total_csv}',
        f'Valid files included: {(audit.status == "INCLUDED").sum()}',
        f'Files excluded above length limit: {(audit.status == "EXCLUDED").sum()}',
        f'Invalid files skipped: {(audit.status == "SKIPPED").sum()}',
        f'Expected conditions present: {complete} / {expected}',
        f'Balanced-only analysis: {args.balanced_only}',
        f'Source FASTA datasets used for analysis: {clean.source_fasta.nunique()}',
        f'Original + nonzero-treatment alignment rows retained: {len(clean)}',
        f'Independent starting candidate pairs: {clean.group_id.nunique()}',
        f'Highest available length: {int(clean.query_length.max())} bp',
        f'Graphs of mutation rate use: {plotted} bp',
        f'Model fixed-effect coefficient estimates saved: {ncoef}',
        '',
        'IMPORTANT:',
        '- Duplicated p=0 treatment files are reduced to one baseline per pair/algorithm.',
        '- p>0 draws with zero actual mutations are retained; these are valid binomial draws.',
        '- Alignment score degradation is NOT biological alignment accuracy.',
        '- Memory is Python-traced peak allocation, NOT total system RAM.',
        '- Each alignment was timed once; coefficients may reflect hardware drift and run order.',
        '- There are three independent source queries per sequence length in the 2026 dataset.',
        '- log_cells and indel rate can be highly correlated; treat adjusted indel terms cautiously.',
        '- Primary model predictors use REQUESTED probability p; realized_rate is also saved for later sensitivity checks.',
        '- Model results are observational associations in a controlled simulation, not proof of causal overhead.',
        '- Some conditions may be incomplete; inspect completeness.csv before comparing groups.',
    ]
    (out / 'ANALYSIS_README.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report[:16]))
    print(f'\nOutputs saved in: {out.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='data/results', help='Root folder with original/substitution/deletion/insertion CSVs')
    parser.add_argument('--max-length', type=int, default=1000, help='Maximum included query length (default 1000)')
    parser.add_argument('--plot-length', type=int, default=None, help='Length for mutation-rate graph, default highest available')
    parser.add_argument('--expected-rates', type=float, nargs='+', default=RATES, help='Mutation probabilities used for this experiment (include 0)')
    parser.add_argument('--expected-lengths', type=int, nargs='+', default=BLOCK_LENGTHS, help='Block lengths used for this experiment')
    parser.add_argument('--out', default='analysis_outputs', help='Analysis output folder')
    parser.add_argument('--cluster-by', choices=('pair', 'reference'), default='pair',
                        help='Fallback regression clusters: 150 candidate pairs (original fit) or 15 references (sensitivity)')
    parser.add_argument('--no-models', action='store_true', help='Only validate, combine and graph the CSVs')
    parser.add_argument('--balanced-only', action='store_true', help='Analyse only source FASTA lengths with every expected rate/type/block condition present')
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error('--max-length must be positive')
    if not all(0 <= p <= 1 for p in args.expected_rates) or not args.expected_lengths or min(args.expected_lengths) < 1:
        parser.error('Expected rates must be in [0,1] and expected block lengths positive')
    try:
        run(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
