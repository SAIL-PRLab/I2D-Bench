#!/usr/bin/env python3
"""
Calculate average metrics for result files and generate a summary CSV report.
Performs modular analysis of TTS benchmark results.
"""

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# Filter warnings from scipy/numpy
warnings.filterwarnings('ignore')

# ==============================================================================
# Constants & Configuration
# ==============================================================================

OBJECTIVE_METRICS = {
    'spk_similarity': {'higher_better': True},
    'utmosv2': {'higher_better': True},
    'dns_overall': {'higher_better': True},
    'cer': {'higher_better': False},
    'whisper_wer': {'higher_better': False},
    'angry_f1': {'higher_better': True},
    'happy_f1': {'higher_better': True},
    'sad_f1': {'higher_better': True},
}

DEFAULT_METRICS = ["spk_similarity", "cer", "utmosv2", "dns_overall"]

# ==============================================================================
# Composite Scoring System
# ==============================================================================

class CompositeScorer:
    """Calculates comprehensive scores based on metric trends across iterations."""

    @staticmethod
    def get_methods() -> List[Tuple[str, Any]]:
        return [
            ('Uniform Weight', CompositeScorer.uniform_weight),
            ('Linear Weight', CompositeScorer.linear_weight),
            ('Exponential Decay', CompositeScorer.exponential_decay),
            ('Normalized AUC', CompositeScorer.normalized_auc),
            ('Peak Stability', CompositeScorer.peak_stability),
        ]

    @staticmethod
    def uniform_weight(values: np.ndarray, **kwargs) -> float:
        return np.nanmean(values)

    @staticmethod
    def linear_weight(values: np.ndarray, **kwargs) -> float:
        """Later iterations have higher weight."""
        valid_mask = ~np.isnan(values)
        if not valid_mask.any(): return np.nan
        
        values = values[valid_mask]
        weights = np.arange(1, len(values) + 1, dtype=float)
        return np.average(values, weights=weights/weights.sum())

    @staticmethod
    def exponential_decay(values: np.ndarray, decay_rate: float = 0.9, **kwargs) -> float:
        """Earlier iterations have higher weight."""
        valid_mask = ~np.isnan(values)
        if not valid_mask.any(): return np.nan
        
        values = values[valid_mask]
        weights = np.array([decay_rate ** i for i in range(len(values))])
        return np.average(values, weights=weights/weights.sum())

    @staticmethod
    def normalized_auc(values: np.ndarray, **kwargs) -> float:
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 2: return np.nan
        
        y = values[valid_mask]
        x = np.arange(len(y))
        
        auc = integrate.trapezoid(y, x)
        x_range = x[-1] - x[0]
        return auc / x_range if x_range > 0 else y[0]

    @staticmethod
    def peak_stability(values: np.ndarray, higher_better: bool = True, **kwargs) -> float:
        valid = values[~np.isnan(values)]
        if len(valid) == 0: return np.nan
        
        std_dev = np.std(valid)
        return (np.max(valid) - std_dev) if higher_better else (np.min(valid) + std_dev)

# ==============================================================================
# Data Processing
# ==============================================================================

class MetricsProcessor:
    """Handles parsing and aggregation of metric files."""

    @staticmethod
    def parse_emo_file(path: Path) -> Dict[str, float]:
        """Parses emotion score files for F1 scores."""
        scores = {}
        target_emotions = ["angry", "happy", "sad"]
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return scores

        lines = content.splitlines()
        start_parsing = False
        for line in lines:
            if "====Score for all====" in line:
                start_parsing = True
                continue
            if start_parsing:
                parts = line.split()
                if not parts: continue
                
                emotion = parts[0]
                if emotion in target_emotions and len(parts) >= 4:
                    try:
                        scores[f"{emotion}_f1"] = float(parts[3])
                    except ValueError: pass
                
                if emotion == "weighted":
                    scores["weighted_f1"] = float(parts[4]) if len(parts) >= 4 else None
                
        return scores

    @staticmethod
    def aggregate_group(files: Dict[str, List[Path]], metrics: List[str]) -> Dict[str, Any]:
        """Computes average metrics for a group of files."""
        accum = {m: {"sum": 0.0, "valid": 0, "skipped": 0} for m in metrics}
        seen_files = 0

        # 1. Process standard result files (json/jsonl)
        for p in files.get('obj', []):
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    seen_files += 1
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                        
                    for m in metrics:
                        MetricsProcessor._accumulate_metric(accum, m, data)

        # 2. Process aggregated emotion files
        for p in files.get('emo', []):
            emo_scores = MetricsProcessor.parse_emo_file(p)
            for m in metrics:
                if m in emo_scores:
                    # Treat pre-calculated score as a single weighted sample or just sum it?
                    # Original logic added it directly. Assuming it represents the whole set.
                    accum[m]["sum"] += emo_scores[m]
                    accum[m]["valid"] += 1

        # Calculate averages
        results = {}
        for m in metrics:
            valid = accum[m]["valid"]
            results[m] = {
                "average": (accum[m]["sum"] / valid) if valid > 0 else None,
                "count": valid
            }
        
        return {"seen": seen_files, "metrics": results}

    @staticmethod
    def _accumulate_metric(accum: Dict, metric: str, data: Dict):
        """Helper to process a single data point."""
        if metric == "wer":
            # Whisper WER calculation
            keys = ["whisper_wer_replace", "whisper_wer_delete", "whisper_wer_insert", "whisper_wer_equal"]
            vals = [data.get(k) for k in keys]
            if all(isinstance(v, (int, float)) for v in vals):
                S, D, I, C = map(float, vals)
                if all(math.isfinite(x) and x >= 0 for x in (S, D, I, C)):
                    errors = S + D + I
                    total = C + S + D
                    if total > 0:
                        accum[metric]["sum"] += errors
                        accum[metric]["valid"] += total
                        return
        else:
            # Standard metrics
            val = data.get(metric)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                accum[metric]["sum"] += float(val)
                accum[metric]["valid"] += 1
                return
        
        accum[metric]["skipped"] += 1

# ==============================================================================
# File Organization
# ==============================================================================

def scan_files(base_dir: Path, output_file: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Scans for result files and groups them by iteration or filename."""
    groups = {}
    
    # helper
    def add(key, type_, p):
        if key not in groups: groups[key] = {'obj': [], 'emo': []}
        groups[key][type_].append(p)

    candidates = sorted(list(base_dir.rglob("*.txt")))
    
    # Filter and group
    output_resolve = output_file.resolve()
    for p in candidates:
        if not p.is_file(): continue
        try:
            if p.resolve() == output_resolve: continue
        except OSError: pass

        # Identify group
        try:
            rel_str = str(p.relative_to(base_dir))
            match = re.search(r"iter_(\d+)", rel_str)
            if match:
                key = f"iter_{match.group(1)}"
            else:
                # Top level files only for non-iter
                if p.parent.resolve() != base_dir.resolve(): continue
                key = p.name
        except ValueError: continue

        # Categorize
        if p.name == "emo_score.txt":
            add(key, 'emo', p)
        else:
            add(key, 'obj', p)
            
    return groups

# ==============================================================================
# Reporting
# ==============================================================================

def generate_csv(metrics: List[str], stats: List[Dict]) -> str:
    """Generates the CSV content with transposed structure and composite scores."""
    
    # 1. Sort stats by iteration
    def get_iter(s):
        match = re.search(r"iter_(\d+)", str(s["path"]))
        return int(match.group(1)) if match else None

    stats.sort(key=lambda s: (0, get_iter(s)) if get_iter(s) is not None else (1, str(s['path'])))

    # 2. Columns: Iterations + Composite Methods
    iter_keys = []
    for s in stats:
        i = get_iter(s)
        iter_keys.append(str(i) if i is not None else s["path"])
    
    comp_methods = CompositeScorer.get_methods()
    headers = ["Metric"] + iter_keys + [name for name, _ in comp_methods]
    
    lines = [",".join(headers)]

    # 3. Rows: Metrics
    for m in metrics:
        row = [m]
        values_for_score = [] # list of (iter, value)

        # Fill iteration values
        for s in stats:
            avg = s["metrics"][m]["average"]
            row.append(f"{avg:.6f}" if isinstance(avg, float) else "N/A")
            
            i = get_iter(s)
            if i is not None and isinstance(avg, float):
                values_for_score.append((i, avg))
        
        # Calculate composite scores
        iter_arr = np.array([v[0] for v in values_for_score])
        val_arr = np.array([v[1] for v in values_for_score])
        
        if len(iter_arr) > 0:
            sort_idx = np.argsort(iter_arr)
            val_arr = val_arr[sort_idx]

        for _, func in comp_methods:
            try:
                if len(val_arr) == 0:
                    row.append("N/A")
                    continue
                
                # Setup kwargs
                kwargs = {}
                if func == CompositeScorer.peak_stability and m in OBJECTIVE_METRICS:
                    kwargs['higher_better'] = OBJECTIVE_METRICS[m]['higher_better']
                
                score = func(val_arr, **kwargs)
                
                row.append(f"{score:.6f}" if isinstance(score, float) and not np.isnan(score) else "N/A")
            except Exception as e:
                print(f"Error calculating composite score for metric {m} with method {func}: {e}")
                row.append("Error")

        lines.append(",".join(row))

    return "\n".join(lines)

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="TTS Benchmark Score Calculator")
    parser.add_argument("--input_dir", required=True, help="Input directory")
    parser.add_argument("--output", help="Output file path (default: input_dir/score_summary.csv)")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to compute")
    parser.add_argument("--metric", help="Single metric override")
    args = parser.parse_args()

    # Setup paths and metrics
    input_path = Path(args.input_dir).expanduser().resolve()
    base_dir = input_path if input_path.is_dir() else input_path.parent
    
    output_path = Path(args.output).expanduser().resolve() if args.output else base_dir / "score_summary.csv"

    metrics = [args.metric] if args.metric else args.metrics
    
    # Auto-add emo metrics if present
    if any(base_dir.rglob("emo_score.txt")):
        for m in ["angry_f1", "happy_f1", "sad_f1", "weighted_f1"]:
            if m not in metrics: metrics.append(m)

    # Drop the legacy emotion_similarity row; it only carried N/A placeholders
    # and the underlying emotion2vec scorer is no longer used.
    if "emotion_similarity" in metrics:
        metrics = [m for m in metrics if m != "emotion_similarity"]

    # Scan and Process
    if input_path.is_file():
        # Edge case: single file input (rarely used but supported)
        files = {'single': {'obj': [input_path], 'emo': []}}
    else:
        files = scan_files(base_dir, output_path)

    stats = []
    print(f"Processing {len(files)} groups form {base_dir}...")
    for key, group_files in files.items():
        res = MetricsProcessor.aggregate_group(group_files, metrics)
        res['path'] = key
        stats.append(res)

    # Generate Report
    csv_content = generate_csv(metrics, stats)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_content + "\n", encoding="utf-8")
    print(f"Summary written to {output_path}")

if __name__ == "__main__":
    main()
