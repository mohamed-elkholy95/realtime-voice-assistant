"""Evaluation metrics."""
import logging
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger(__name__)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (simplified)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words: return 0.0 if not hyp_words else 1.0
    # Count word-level differences (substitutions, insertions, deletions)
    from collections import Counter
    ref_counts = Counter(ref_words)
    hyp_counts = Counter(hyp_words)
    edits = sum(abs(ref_counts[w] - hyp_counts.get(w, 0)) for w in set(ref_words) | set(hyp_words))
    return min(edits / len(ref_words), 1.0)


def generate_report(metrics: Dict[str, Any]) -> str:
    lines = ["# Voice Assistant — Evaluation Report", "",
             "| Metric | Value |", "|--------|-------|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
