#!/usr/bin/env python3
"""
I2D Benchmark Dashboard

Aggregates objective metrics across synthesis iterations of zero-shot TTS
models, following the I2D (Iterate to Differentiate) framework:

    Shen et al., "Iterate to Differentiate: Enhancing Discriminability and
    Reliability in Zero-Shot TTS Evaluation", 2026.
    https://arxiv.org/pdf/2603.24430

Features:
  - Per-iteration metric values for each model (trajectories)
  - Aggregated scores via mean across synthesis iterations
  - System-level rankings (1 = best, lower is better)
  - Streamlit dashboard

Usage:
    streamlit run utils/comprehensive_analysis.py
"""

import csv
import os
import json
import re
import warnings
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings('ignore')

# ============================================================================
# I2D configuration
# ============================================================================

# Datasets (paper Section 3.2)
DATASETS = {
    'Seed-TTS-Eval': {
        'path': 'results/Seed-Eval/test_zh',
        'language': 'zh',
    },
    'LibriTTS': {
        'path': 'results/LibriTTS/test_clean',
        'language': 'en',
    },
    'CV3-Eval': {
        'path': 'results/CV3-Eval/emotion_zeroshot',
        'language': 'multi',
    },
}

# Objective metrics (paper Table 1)
OBJECTIVE_METRICS = {
    'spk_similarity': {'higher_better': True,  'display_name': 'SIM (Speaker Similarity)', 'paper': 'SIM'},
    'utmosv2':        {'higher_better': True,  'display_name': 'UTMOSv2',                  'paper': 'UTMOSv2'},
    'dns_overall':    {'higher_better': True,  'display_name': 'DNSMOS',                   'paper': 'DNSMOS'},
    'cer':            {'higher_better': False, 'display_name': 'CER (↓)',                  'paper': 'CER'},
    'wer':            {'higher_better': False, 'display_name': 'WER (↓)',                  'paper': 'WER'},
    'angry_f1':       {'higher_better': True,  'display_name': 'Angry F1',                 'paper': 'EMO F1'},
    'happy_f1':       {'higher_better': True,  'display_name': 'Happy F1',                 'paper': 'EMO F1'},
    'sad_f1':         {'higher_better': True,  'display_name': 'Sad F1',                   'paper': 'EMO F1'},
    'weighted_f1':    {'higher_better': True,  'display_name': 'Weighted Avg F1',          'paper': 'EMO F1'},
}

# Consistent color per metric, used across all plots
METRIC_COLORS = {
    'spk_similarity': '#1f77b4',
    'utmosv2':        '#2ca02c',
    'dns_overall':    '#9467bd',
    'cer':            '#d62728',
    'wer':            '#ff7f0e',
    'angry_f1':       '#e377c2',
    'happy_f1':       '#17becf',
    'sad_f1':         '#bcbd22',
    'weighted_f1':    '#8c564b',
}

RESULTS_ROOT = 'results'
OUTPUT_DIR = 'analysis_results'


# ============================================================================
# Data loading
# ============================================================================

def parse_score_summary_txt(file_path: str) -> Optional[pd.DataFrame]:
    """Parse LibriTTS-style score_summary.txt

    Format: iter_X.gpu.txt  spk_sim  wer  utmosv2  dns  count
    """
    if not os.path.exists(file_path):
        return None
    data = []
    pattern = re.compile(
        r'iter_(\d+)\.gpu\.txt\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
    )
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            data.append({
                'iteration':      int(m.group(1)),
                'spk_similarity': float(m.group(2)),
                'wer':            float(m.group(3)),
                'utmosv2':        float(m.group(4)),
                'dns_overall':    float(m.group(5)),
                'sample_count':   int(m.group(6)),
            })
    return pd.DataFrame(data) if data else None


def parse_score_summary_csv(file_path: str) -> Optional[pd.DataFrame]:
    """Parse Seed-TTS-Eval / CV3-Eval-style score_summary.csv

    Expected format:
        Metric,1,2,3,...
        <metric_name>,<v1>,<v2>,...
    Returns long format: one row per iteration.

    Defensive: csv.reader-based to tolerate malformed second-header rows
    (e.g. CV3-Eval may have a `angry_f1,happy_f1,...` descriptive row that
    pd.read_csv's field-count tolerance would misalign into all rows).
    """
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', newline='') as f:
            raw = [r for r in csv.reader(f) if r]
        if len(raw) < 2:
            return None
        header = raw[0]
        if 'Metric' not in header:
            return None
        metric_idx = header.index('Metric')
        iter_indices = {
            int(c): header.index(c)
            for c in (str(i) for i in range(1, 11))
            if c in header
        }
        if not iter_indices:
            return None
        out = []
        for iter_num, idx in sorted(iter_indices.items()):
            row = {'iteration': iter_num}
            for data_row in raw[1:]:
                if metric_idx >= len(data_row) or idx >= len(data_row):
                    continue
                name = data_row[metric_idx].strip()
                if not name:
                    continue
                try:
                    row[name] = float(data_row[idx])
                except ValueError:
                    continue
            out.append(row)
        return pd.DataFrame(out) if out else None
    except Exception as e:
        print(f'parse_score_summary_csv failed: {e}')
        return None


def parse_score_summary(model_path: str) -> Optional[pd.DataFrame]:
    """Auto-detect and parse score_summary file for a model directory."""
    csv_path = os.path.join(model_path, 'score_summary.csv')
    if os.path.exists(csv_path):
        return parse_score_summary_csv(csv_path)
    txt_path = os.path.join(model_path, 'score_summary.txt')
    if os.path.exists(txt_path):
        return parse_score_summary_txt(txt_path)
    return None


@st.cache_data
def load_objective_data(dataset_path: str) -> Dict[str, pd.DataFrame]:
    """Load per-utterance iter_X.gpu.txt for all models in a dataset."""
    results: Dict[str, pd.DataFrame] = {}
    if not os.path.exists(dataset_path):
        return results
    for model_name in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model_name)
        if not os.path.isdir(model_path):
            continue
        records = []
        for iter_num in range(1, 11):
            iter_file = os.path.join(model_path, f'iter_{iter_num}.gpu.txt')
            if not os.path.exists(iter_file):
                continue
            with open(iter_file, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        rec['iteration'] = iter_num
                        rec['model'] = model_name
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
        if records:
            results[model_name] = pd.DataFrame(records)
    return results


@st.cache_data
def load_summary_data(dataset_path: str) -> pd.DataFrame:
    """Load aggregated score_summary files for all models."""
    all_data = []
    if not os.path.exists(dataset_path):
        return pd.DataFrame()
    for model_name in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model_name)
        if not os.path.isdir(model_path):
            continue
        summary_df = parse_score_summary(model_path)
        if summary_df is not None:
            summary_df['model'] = model_name
            all_data.append(summary_df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def compute_model_statistics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fallback: compute per-iteration mean from per-utterance data."""
    stats = []
    for model_name, df in data.items():
        for iteration in sorted(df['iteration'].unique()):
            iter_df = df[df['iteration'] == iteration]
            row = {
                'model': model_name,
                'iteration': int(iteration),
                'sample_count': len(iter_df),
            }
            for metric in OBJECTIVE_METRICS.keys():
                row[metric] = float(iter_df[metric].mean()) if metric in iter_df.columns else np.nan
            stats.append(row)
    return pd.DataFrame(stats)


# ============================================================================
# Aggregation (paper Eqs. 1-4)
# ============================================================================

def compute_aggregated_scores(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Average per-iteration metric values per model (mean across iterations).

    Returns wide format: one row per model, one column per metric.
    sample_count is summed across iterations.
    """
    if stats_df.empty:
        return pd.DataFrame()
    metric_cols = [m for m in OBJECTIVE_METRICS.keys() if m in stats_df.columns]
    if not metric_cols:
        return pd.DataFrame()
    grouped = stats_df.groupby('model', sort=False)
    agg = grouped[metric_cols].mean().reset_index()
    if 'sample_count' in stats_df.columns:
        counts = grouped['sample_count'].sum().reset_index()
        agg = agg.merge(counts, on='model', how='left')
    return agg


def compute_model_rankings(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Rank each model on each metric (1 = best, direction from higher_better)."""
    rankings = []
    for metric in OBJECTIVE_METRICS.keys():
        if metric not in metrics_df.columns:
            continue
        ranked = metrics_df[['model', metric]].dropna(subset=[metric])
        if ranked.empty:
            continue
        higher_better = OBJECTIVE_METRICS[metric]['higher_better']
        ranked = ranked.sort_values(metric, ascending=not higher_better)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rankings.append({
                'model':  row['model'],
                'metric': metric,
                'value':  row[metric],
                'rank':   rank,
            })
    return pd.DataFrame(rankings)


# ============================================================================
# Visualization
# ============================================================================

def plot_metric_comparison(
    stats_df: pd.DataFrame,
    metric: str,
    iteration: int = 1,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[plt.Figure]:
    """Single-metric horizontal bar chart at a fixed iteration depth."""
    iter_df = stats_df[stats_df['iteration'] == iteration].copy()
    if metric not in iter_df.columns:
        return None
    config = OBJECTIVE_METRICS.get(metric, {'higher_better': True, 'display_name': metric})
    iter_df = iter_df.dropna(subset=[metric]).sort_values(
        metric, ascending=not config['higher_better']
    )
    if iter_df.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    color = METRIC_COLORS.get(metric, '#1f77b4')
    bars = ax.barh(iter_df['model'], iter_df[metric], color=color, alpha=0.85)
    arrow = '↑ better' if config['higher_better'] else '↓ better'
    ax.set_xlabel(config['display_name'], fontsize=11)
    ax.set_title(f"{config['display_name']} at Iter {iteration}  ({arrow})", fontsize=13, pad=12)
    ax.invert_yaxis()
    xmax = iter_df[metric].max()
    for bar, value in zip(bars, iter_df[metric]):
        ax.text(bar.get_width() + abs(xmax) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{value:.4f}', va='center', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig


def plot_iteration_trend(
    stats_df: pd.DataFrame,
    metric: str,
    models: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[plt.Figure]:
    """Per-iteration metric trajectory across synthesis rounds.

    Paper Section 5.2 — stronger models degrade more slowly (differential
    degradation), so the spread across curves widens with iteration depth.
    """
    if metric not in stats_df.columns:
        return None
    if models is None:
        models = list(stats_df['model'].unique())

    fig, ax = plt.subplots(figsize=figsize)
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(models))))
    for i, model in enumerate(models):
        model_df = stats_df[stats_df['model'] == model].sort_values('iteration')
        ax.plot(model_df['iteration'], model_df[metric],
                marker='o', linewidth=2, markersize=6,
                color=palette[i % len(palette)], label=model)
    config = OBJECTIVE_METRICS.get(metric, {'display_name': metric, 'higher_better': True})
    arrow = '↑ better' if config['higher_better'] else '↓ better'
    ax.set_xlabel('Iteration depth', fontsize=11)
    ax.set_ylabel(config['display_name'], fontsize=11)
    ax.set_title(f"{config['display_name']} Trajectory  ({arrow})", fontsize=13, pad=12)
    max_iter = int(stats_df['iteration'].max())
    ax.set_xticks(range(1, max_iter + 1))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_multi_metric_bar(
    stats_df: pd.DataFrame,
    metrics: List[str],
    iteration: int = 1,
    figsize: Tuple[int, int] = (14, 8),
) -> Optional[plt.Figure]:
    """Min-max normalised multi-metric comparison at a fixed iteration depth."""
    iter_df = stats_df[stats_df['iteration'] == iteration].copy()
    valid_metrics = [m for m in metrics if m in iter_df.columns]
    if not valid_metrics:
        return None

    normalized_df = iter_df.copy()
    for metric in valid_metrics:
        cfg = OBJECTIVE_METRICS.get(metric, {'higher_better': True})
        col = iter_df[metric].astype(float)
        lo, hi = col.min(), col.max()
        if hi > lo:
            norm = (col - lo) / (hi - lo)
            if not cfg['higher_better']:
                norm = 1 - norm
        else:
            norm = pd.Series(0.5, index=iter_df.index)
        normalized_df[f'{metric}_norm'] = norm
    norm_cols = [f'{m}_norm' for m in valid_metrics]
    normalized_df['overall'] = normalized_df[norm_cols].mean(axis=1)
    normalized_df = normalized_df.sort_values('overall', ascending=False)

    models = normalized_df['model'].values
    x = np.arange(len(models))
    width = 0.8 / len(valid_metrics)
    fig, ax = plt.subplots(figsize=figsize)
    for i, metric in enumerate(valid_metrics):
        offset = (i - len(valid_metrics) / 2 + 0.5) * width
        color = METRIC_COLORS.get(metric, f'C{i}')
        display = OBJECTIVE_METRICS.get(metric, {}).get('display_name', metric)
        ax.bar(x + offset, normalized_df[f'{metric}_norm'], width,
               label=display, color=color, alpha=0.8)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Normalized score (0–1, higher = better)', fontsize=11)
    ax.set_title(f'Multi-Metric Comparison at Iter {iteration}', fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


# ============================================================================
# Helpers
# ============================================================================

def format_dataframe(df: pd.DataFrame, decimal_places: int = 4) -> pd.DataFrame:
    """Format numeric columns; rank columns are kept as integers."""
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if df_formatted[col].dtype in ['float64', 'float32']:
            if 'rank' in col.lower():
                df_formatted[col] = df_formatted[col].astype('Int64')
            else:
                df_formatted[col] = df_formatted[col].round(decimal_places)
    return df_formatted


def export_analysis_results(
    stats_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> bool:
    os.makedirs(output_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(output_dir, 'model_statistics.csv'), index=False)
    rankings_df.to_csv(os.path.join(output_dir, 'model_rankings.csv'), index=False)
    return True


def _rename_to_display(df: pd.DataFrame) -> pd.DataFrame:
    """Rename metric columns to their paper-style display names."""
    if df.empty:
        return df
    mapping = {m: cfg['display_name'] for m, cfg in OBJECTIVE_METRICS.items() if m in df.columns}
    return df.rename(columns=mapping)


def _ordered_metrics(metrics) -> List[str]:
    """Return metrics in OBJECTIVE_METRICS insertion order, filtered to the given set."""
    return [m for m in OBJECTIVE_METRICS.keys() if m in metrics]


# ============================================================================
# Streamlit styling
# ============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
        .i2d-header {
            padding: 1.25rem 1.5rem;
            background: linear-gradient(135deg, #1f2c4c 0%, #2d4373 100%);
            color: #fff;
            border-radius: 0.6rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }
        .i2d-header h1 { color: #fff !important; margin: 0; font-size: 1.6rem; }
        .i2d-header p  { color: #cfd8ec; margin: 0.35rem 0 0; font-size: 0.9rem; }
        .i2d-header a  { color: #8ab4ff; text-decoration: none; }
        .stDataFrame { font-size: 0.88rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; font-weight: 500; }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Streamlit main UI
# ============================================================================

def main():
    st.set_page_config(
        page_title='I2D Benchmark',
        page_icon='🎙️',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    inject_custom_css()

    # ---- Header band ----
    st.markdown("""
    <div class="i2d-header">
        <h1>🎙️ I2D Benchmark </h1>
        <p>Iterate to Differentiate: Enhancing Discriminability and Reliability in Zero-Shot TTS Evaluation
           &nbsp;·&nbsp; <a href="https://arxiv.org/pdf/2603.24430" target="_blank">Shen et al., 2026 (arXiv:2603.24430)</a></p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.header('⚙️ Configuration')
        st.divider()
        dataset_name = st.selectbox(
            'Dataset',
            list(DATASETS.keys()),
            format_func=lambda x: f"{x}",
        )

        iteration = st.slider(
            'Iteration depth (per-iteration views)',
            1, 10, 1,
            help='Iteration N for the "Per-Iteration Comparison" tab. '
                 'Rankings always use the aggregated score over all iterations.',
        )
        st.divider()
        if st.button('📥 Export results'):
            st.session_state['export_triggered'] = True

    dataset_config = DATASETS[dataset_name]

    # ---- Load data ----
    with st.spinner('Loading dataset...'):
        summary_df = load_summary_data(dataset_config['path'])
        objective_data = load_objective_data(dataset_config['path'])

    if summary_df.empty and not objective_data:
        st.error(f"Could not load dataset: **{dataset_name}** (path: `{dataset_config['path']}`)")
        return

    if not summary_df.empty:
        stats_df = summary_df
        source = 'score_summary'
    else:
        stats_df = compute_model_statistics(objective_data)
        source = 'per-utterance fallback'

    available_metrics = [m for m in OBJECTIVE_METRICS.keys() if m in stats_df.columns]
    if not available_metrics:
        st.error('No supported objective metrics found in this dataset.')
        return

    # ---- Top-line KPIs ----
    n_models = stats_df['model'].nunique()
    n_iters = int(stats_df['iteration'].max())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Dataset', dataset_name)
    k2.metric('Models', n_models)
    k3.metric('Iterations (N)', n_iters)
    k4.metric('Metrics', len(available_metrics))
    st.divider()

    # ---- Compute aggregation & rankings ----
    agg_df = compute_aggregated_scores(stats_df)
    if agg_df.empty:
        st.error('Aggregated scores are empty; check data integrity.')
        return
    rankings_df = compute_model_rankings(agg_df)

    # ---- Tabs ----
    tabs = st.tabs([
        '🏆 System-Level Rankings',
        '📈 Iteration Trajectories',
        '🎯 Per-Iteration Comparison',
    ])

    # ===== Tab 1: System-Level Rankings =====
    with tabs[0]:
        st.header('System-Level Rankings')
        st.caption(
            f"Rankings derived from the mean of per-iteration metric values "
            f"over {n_iters} synthesis iterations. Lower rank is better. "
            f"Models are sorted by their mean rank across all metrics."
        )

        st.subheader('Rank Table')
        if not rankings_df.empty:
            rank_pivot = rankings_df.pivot(index='model', columns='metric', values='rank')
            rank_pivot = rank_pivot[_ordered_metrics(rank_pivot.columns)]
            rank_pivot['Mean Rank'] = rank_pivot.mean(axis=1).round(2)
            rank_pivot = rank_pivot.sort_values('Mean Rank')
            rank_pivot = _rename_to_display(rank_pivot)
            st.dataframe(format_dataframe(rank_pivot), use_container_width=True)
        else:
            st.info('No ranking data available.')

        st.subheader('Aggregated Scores (mean across iterations)')
        if not agg_df.empty:
            display_agg = _rename_to_display(agg_df).set_index('model')
            st.dataframe(format_dataframe(display_agg), use_container_width=True)
        else:
            st.info('No aggregated data available.')

        with st.expander('Detailed rankings (long format)', expanded=False):
            if not rankings_df.empty:
                detail = rankings_df.copy()
                detail['metric'] = detail['metric'].map(
                    lambda m: OBJECTIVE_METRICS.get(m, {}).get('display_name', m)
                )
                st.dataframe(format_dataframe(detail), use_container_width=True)

    # ===== Tab 2: Iteration Trajectories =====
    with tabs[1]:
        st.header('Iteration Trajectories')
        st.caption(
            "Per-iteration metric values across synthesis rounds. "
            "Robust models degrade slowly; weaker models collapse early — "
            "the I2D *differential degradation* principle (paper §5.2)."
        )

        ctrl_col, plot_col = st.columns([1, 3])
        with ctrl_col:
            selected_metric = st.selectbox(
                'Metric',
                available_metrics,
                format_func=lambda x: OBJECTIVE_METRICS[x]['display_name'],
            )
            all_models = list(stats_df['model'].unique())
            selected_models = st.multiselect(
                'Models (empty = all)', all_models, default=[],
            )
        with plot_col:
            fig_trend = plot_iteration_trend(
                stats_df, selected_metric, models=selected_models or None,
            )
            if fig_trend:
                st.pyplot(fig_trend)
                plt.close(fig_trend)

    # ===== Tab 3: Per-Iteration Comparison =====
    with tabs[2]:
        st.header(f'Per-Iteration Comparison at depth N = {iteration}')
        st.caption(
            "At fixed iteration depth, compare models on one or several metrics. "
            "For the multi-metric view, all metrics are min-max normalised to [0, 1] "
            "with direction flipped for lower-better metrics."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Single-Metric Bar')
            compare_metric = st.selectbox(
                'Metric',
                available_metrics,
                format_func=lambda x: OBJECTIVE_METRICS[x]['display_name'],
                key='compare_metric',
            )
            fig_compare = plot_metric_comparison(stats_df, compare_metric, iteration)
            if fig_compare:
                st.pyplot(fig_compare)
                plt.close(fig_compare)
        with col2:
            st.subheader('Multi-Metric Normalized Comparison')
            # Default: content (CER/WER), SIM, UTMOSv2, DNSMOS — paper's four
            default_multi = [m for m in ['spk_similarity', 'utmosv2', 'dns_overall'] if m in available_metrics]
            for c in ['cer', 'wer']:
                if c in available_metrics:
                    default_multi.append(c)
                    break
            multi_metrics = st.multiselect(
                'Metrics',
                available_metrics,
                default=default_multi,
                format_func=lambda x: OBJECTIVE_METRICS[x]['display_name'],
                key='multi_metrics',
            )
            if multi_metrics:
                fig_multi = plot_multi_metric_bar(stats_df, _ordered_metrics(multi_metrics), iteration)
                if fig_multi:
                    st.pyplot(fig_multi)
                    plt.close(fig_multi)

    # ---- Export ----
    if st.session_state.get('export_triggered', False):
        with st.spinner('Exporting...'):
            success = export_analysis_results(
                stats_df, rankings_df,
                output_dir=os.path.join(OUTPUT_DIR, dataset_name),
            )
            if success:
                st.sidebar.success(f"✅ Exported to `{OUTPUT_DIR}/{dataset_name}/`")
            else:
                st.sidebar.error('❌ Export failed')
        st.session_state['export_triggered'] = False


if __name__ == '__main__':
    main()
