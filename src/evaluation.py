"""Evaluation metrics for the voice assistant pipeline.

This module provides comprehensive evaluation metrics used to assess the
quality of Speech-to-Text systems, intent classifiers, and the overall
pipeline performance.

Educational Context:
    Word Error Rate (WER) is the standard metric for evaluating ASR systems.
    It measures the minimum number of edits (substitutions, insertions, deletions)
    needed to transform the hypothesis into the reference, normalized by the
    number of words in the reference.

    The edit distance algorithm used here (Levenshtein distance with dynamic
    programming) is a foundational algorithm in computer science with
    applications in spell checking, DNA sequence alignment, and diff tools.

    WER = (S + D + I) / N
    Where: S = substitutions, D = deletions, I = insertions, N = reference words

    A WER of 0% means perfect transcription. Human conversational speech
    typically has WERs of 4-5%, while ASR systems range from 5-30% depending
    on conditions (clean speech, noise, accents, domain).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def levenshtein_distance(
    sequence_a: List[str],
    sequence_b: List[str],
) -> Tuple[int, int, int, int]:
    """Compute Levenshtein edit distance between two sequences using dynamic programming.

    The Levenshtein distance counts the minimum number of single-element
    operations (insertions, deletions, substitutions) required to transform
    sequence_a into sequence_b.

    Dynamic Programming Approach:
        We build a 2D table where dp[i][j] represents the edit distance
        between the first i elements of sequence_a and the first j elements
        of sequence_b.

        Recurrence relation:
            dp[i][j] = min(
                dp[i-1][j] + 1,              # deletion from sequence_a
                dp[i][j-1] + 1,              # insertion into sequence_a
                dp[i-1][j-1] + cost          # substitution (cost=0 if match)
            )

        where cost = 0 if sequence_a[i-1] == sequence_b[j-1], else 1.

    Time complexity: O(m * n) where m = len(a), n = len(b)
    Space complexity: O(m * n) for the full table (can be optimized to O(min(m,n)))

    Args:
        sequence_a: First sequence of tokens (e.g., reference words).
        sequence_b: Second sequence of tokens (e.g., hypothesis words).

    Returns:
        A tuple of (distance, substitutions, deletions, insertions):
            - distance: Total edit distance (S + D + I).
            - substitutions: Number of substitution operations.
            - deletions: Number of deletion operations.
            - insertions: Number of insertion operations.

    Example:
        >>> levenshtein_distance(["hello", "world"], ["hello", "there"])
        (1, 1, 0, 0)
        >>> levenshtein_distance(["a", "b", "c"], ["a", "x", "c"])
        (1, 1, 0, 0)
        >>> levenshtein_distance([], ["hello"])
        (1, 0, 0, 1)
    """
    len_a = len(sequence_a)
    len_b = len(sequence_b)

    # Handle edge cases
    if len_a == 0:
        return len_b, 0, 0, len_b  # All insertions
    if len_b == 0:
        return len_a, 0, len_a, 0  # All deletions

    # Initialize DP table of size (len_a + 1) x (len_b + 1)
    # dp[i][j] = edit distance between a[:i] and b[:j]
    dp = np.zeros((len_a + 1, len_b + 1), dtype=np.int32)

    # Base cases: transforming from/to empty sequence
    # dp[i][0] = i (delete all elements from sequence_a)
    for i in range(len_a + 1):
        dp[i][ 0] = i
    # dp[0][j] = j (insert all elements from sequence_b)
    for j in range(len_b + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if sequence_a[i - 1] == sequence_b[j - 1]:
                # Characters match — no edit needed, carry over diagonal
                substitution_cost = 0
            else:
                # Characters differ — substitution needed
                substitution_cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + substitution_cost,  # substitution or match
            )

    # Backtrack through the DP table to count operation types
    # This lets us report S, D, I separately (not just total distance)
    substitutions = 0
    deletions = 0
    insertions = 0

    i, j = len_a, len_b
    while i > 0 or j > 0:
        if i > 0 and j > 0 and sequence_a[i - 1] == sequence_b[j - 1]:
            # Match — move diagonally
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # Insertion
            insertions += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion
            deletions += 1
            i -= 1
        else:
            # Shouldn't reach here, but safety fallback
            i -= 1
            j -= 1

    total_distance = int(dp[len_a][len_b])
    return total_distance, substitutions, deletions, insertions


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) using dynamic programming.

    WER measures the accuracy of a speech recognition system by comparing
    its output (hypothesis) against the ground truth (reference).

    Formula: WER = (S + D + I) / N
    Where:
        S = number of word substitutions (wrong word chosen)
        D = number of word deletions (word missed entirely)
        I = number of word insertions (extra word hallucinated)
        N = number of words in the reference

    This implementation uses real edit distance (Levenshtein) rather
    than the simplified Counter-based approach, providing accurate WER
    values that account for word order.

    Args:
        reference: The ground truth transcription text.
        hypothesis: The ASR system's transcription output.

    Returns:
        WER as a float in [0, 1]. Lower is better.
        - 0.0 = perfect match
        - 1.0 = every word is wrong (or reference is empty but hypothesis isn't)

    Example:
        >>> compute_wer("hello world", "hello world")
        0.0
        >>> compute_wer("hello world", "hello there")  # 1 substitution
        0.5
        >>> compute_wer("the cat sat", "the cat sat on the mat")  # 3 insertions
        1.0
        >>> compute_wer("", "hello")
        1.0
    """
    # Normalize and tokenize: lowercase, split on whitespace
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    # Edge case: empty reference
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Edge case: both empty
    if not hyp_words:
        return 1.0  # All reference words were deleted

    # Compute edit distance using dynamic programming
    distance, _, _, _ = levenshtein_distance(ref_words, hyp_words)

    # WER = total edits / reference length
    wer = distance / len(ref_words)
    return min(wer, 1.0)  # Cap at 1.0


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) using dynamic programming.

    CER is the character-level equivalent of WER. It measures the
    edit distance at the character level, which is useful for:
    - Languages without clear word boundaries (e.g., Chinese, Japanese)
    - Evaluating fine-grained transcription errors
    - Assessing proper noun / named entity recognition

    Formula: CER = (S + D + I) / N
    Where N = number of characters in the reference.

    Args:
        reference: The ground truth text.
        hypothesis: The predicted text.

    Returns:
        CER as a float in [0, 1]. Lower is better.

    Example:
        >>> compute_cer("hello", "hello")
        0.0
        >>> compute_cer("hello", "helo")  # 1 deletion
        0.2
        >>> compute_cer("cat", "bat")  # 1 substitution
        1/3  # approximately 0.333
    """
    ref_chars = list(reference.lower().strip())
    hyp_chars = list(hypothesis.lower().strip())

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    if not hyp_chars:
        return 1.0

    distance, _, _, _ = levenshtein_distance(ref_chars, hyp_chars)
    cer = distance / len(ref_chars)
    return min(cer, 1.0)


def compute_intent_accuracy(
    references: List[str],
    predictions: List[str],
) -> float:
    """Compute intent classification accuracy.

    Accuracy is the simplest classification metric: the fraction of
    predictions that exactly match the reference labels.

    For more nuanced evaluation, consider:
    - **Precision/Recall/F1**: Per-intent performance (handles class imbalance)
    - **Confusion Matrix**: Shows which intents get confused with each other
    - **Top-k Accuracy**: Whether the correct intent is in the top-k predictions

    Args:
        references: List of true intent labels.
        predictions: List of predicted intent labels.

    Returns:
        Accuracy as a float in [0, 1].

    Raises:
        ValueError: If the lengths of references and predictions don't match.

    Example:
        >>> compute_intent_accuracy(["weather", "time", "music"], ["weather", "time", "general"])
        0.666...
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs "
            f"{len(predictions)} predictions"
        )

    if not references:
        return 0.0

    correct = sum(1 for ref, pred in zip(references, predictions) if ref == pred)
    return correct / len(references)


def compute_confusion_matrix(
    references: List[str],
    predictions: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute a confusion matrix for intent classification evaluation.

    A confusion matrix is a table that visualizes classifier performance
    by showing how often each true label was predicted as each possible
    label. It reveals:

    - **True Positives (diagonal)**: Correctly classified instances
    - **False Positives (column sums minus diagonal)**: Instances wrongly
      assigned to this class
    - **False Negatives (row sums minus diagonal)**: Instances of this
      class that were missed

    From the confusion matrix, per-class precision, recall, and F1 score
    can be derived:

    - **Precision** = TP / (TP + FP) — "Of all predictions for this class,
      how many were correct?" High precision means few false alarms.
    - **Recall** = TP / (TP + FN) — "Of all actual instances of this class,
      how many did we find?" High recall means few missed detections.
    - **F1 Score** = 2 * (precision * recall) / (precision + recall) —
      Harmonic mean of precision and recall, balancing both concerns.

    Args:
        references: List of ground-truth labels.
        predictions: List of predicted labels (same length as references).
        labels: Optional ordered list of label names. If None, labels are
            inferred from the union of references and predictions, sorted
            alphabetically.

    Returns:
        Dictionary containing:
            - 'matrix': 2D list (n_labels × n_labels) of counts.
            - 'labels': Ordered list of label names (row/column headers).
            - 'per_class': Dict mapping each label to precision, recall, f1.
            - 'macro_precision': Macro-averaged precision across all classes.
            - 'macro_recall': Macro-averaged recall across all classes.
            - 'macro_f1': Macro-averaged F1 score across all classes.
            - 'accuracy': Overall classification accuracy.

    Raises:
        ValueError: If references and predictions have different lengths.

    Example:
        >>> refs = ["weather", "weather", "time", "music", "time"]
        >>> preds = ["weather", "time", "time", "music", "weather"]
        >>> result = compute_confusion_matrix(refs, preds)
        >>> result['accuracy']
        0.6
        >>> result['per_class']['weather']['precision']  # 1 correct / 2 predicted
        0.5
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs "
            f"{len(predictions)} predictions"
        )

    if not references:
        return {
            "matrix": [],
            "labels": [],
            "per_class": {},
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "accuracy": 0.0,
        }

    # Determine label ordering
    if labels is None:
        labels = sorted(set(references) | set(predictions))

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n_labels = len(labels)

    # Build the confusion matrix
    matrix = [[0] * n_labels for _ in range(n_labels)]
    for ref, pred in zip(references, predictions):
        ref_idx = label_to_idx.get(ref)
        pred_idx = label_to_idx.get(pred)
        if ref_idx is not None and pred_idx is not None:
            matrix[ref_idx][pred_idx] += 1

    # Compute per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    total_correct = 0

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        total_correct += tp

        # FP = other classes predicted as this class (column sum minus diagonal)
        fp = sum(matrix[row][i] for row in range(n_labels)) - tp
        # FN = this class predicted as other classes (row sum minus diagonal)
        fn = sum(matrix[i][col] for col in range(n_labels)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(matrix[i]),  # Total actual instances of this class
        }

    # Macro-averaged metrics (unweighted mean across classes)
    n_active = sum(1 for m in per_class.values() if m["support"] > 0)
    macro_precision = (
        sum(m["precision"] for m in per_class.values() if m["support"] > 0) / n_active
        if n_active > 0
        else 0.0
    )
    macro_recall = (
        sum(m["recall"] for m in per_class.values() if m["support"] > 0) / n_active
        if n_active > 0
        else 0.0
    )
    macro_f1 = (
        sum(m["f1"] for m in per_class.values() if m["support"] > 0) / n_active
        if n_active > 0
        else 0.0
    )

    accuracy = total_correct / len(references)

    return {
        "matrix": matrix,
        "labels": labels,
        "per_class": per_class,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
    }


def benchmark_latency(
    fn,
    num_runs: int = 10,
    warmup_runs: int = 2,
) -> Dict[str, float]:
    """Benchmark the latency of a function call.

    Measures execution time with warmup runs (to account for JIT compilation,
    cache warming, etc.) and returns descriptive statistics.

    Args:
        fn: Callable to benchmark. Should take no arguments.
            Use functools.partial or lambda to add arguments.
        num_runs: Number of timed runs for computing statistics.
        warmup_runs: Number of untimed warmup runs.

    Returns:
        Dictionary with keys:
            - 'mean_ms': Mean execution time in milliseconds.
            - 'std_ms': Standard deviation in milliseconds.
            - 'min_ms': Minimum execution time.
            - 'max_ms': Maximum execution time.
            - 'median_ms': Median execution time.
            - 'num_runs': Number of timed runs.

    Example:
        >>> import time
        >>> def slow(): time.sleep(0.01)
        >>> result = benchmark_latency(slow, num_runs=5)
        >>> result['mean_ms'] >= 10.0
        True
    """
    # Warmup runs — not measured, but important for consistent benchmarks
    for _ in range(warmup_runs):
        fn()

    # Timed runs
    latencies: List[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000.0  # Convert to ms
        latencies.append(elapsed)

    latencies_array = np.array(latencies)

    return {
        "mean_ms": round(float(np.mean(latencies_array)), 3),
        "std_ms": round(float(np.std(latencies_array)), 3),
        "min_ms": round(float(np.min(latencies_array)), 3),
        "max_ms": round(float(np.max(latencies_array)), 3),
        "median_ms": round(float(np.median(latencies_array)), 3),
        "num_runs": num_runs,
    }


def generate_report(
    metrics: Dict[str, Any],
    include_sections: bool = True,
) -> str:
    """Generate a comprehensive evaluation report in Markdown format.

    The report includes:
    - Header and timestamp
    - Metrics table
    - Interpretation guidelines
    - Optional detailed analysis sections

    Args:
        metrics: Dictionary of metric names to values.
            Values can be numbers, strings, or nested dicts.
        include_sections: Whether to include educational sections
            explaining the metrics.

    Returns:
        Markdown-formatted evaluation report string.

    Example:
        >>> report = generate_report({"wer": 0.15, "cer": 0.08})
        >>> "# Voice Assistant" in report
        True
    """
    import datetime

    lines: List[str] = []
    lines.append("# Voice Assistant — Evaluation Report")
    lines.append(f"\n**Generated:** {datetime.datetime.now().isoformat()}")
    lines.append("")

    # Metrics table
    lines.append("## Metrics Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            # Format floats nicely — percentages for rates, ms for latency
            if "wer" in metric_name.lower() or "cer" in metric_name.lower() \
                    or "accuracy" in metric_name.lower():
                lines.append(f"| {metric_name} | {metric_value:.2%} |")
            elif "ms" in metric_name.lower() or "latency" in metric_name.lower() \
                    or "time" in metric_name.lower():
                lines.append(f"| {metric_name} | {metric_value:.2f} ms |")
            else:
                lines.append(f"| {metric_name} | {metric_value:.4f} |")
        elif isinstance(metric_value, dict):
            lines.append(f"| {metric_name} | {metric_value} |")
        else:
            lines.append(f"| {metric_name} | {metric_value} |")

    # Educational sections
    if include_sections:
        lines.append("")
        lines.append("## Understanding the Metrics")
        lines.append("")
        lines.append("### Word Error Rate (WER)")
        lines.append("- Measures the accuracy of speech-to-text transcription.")
        lines.append("- Calculated as: (substitutions + deletions + insertions) / reference words")
        lines.append("- **0%** = perfect, **<10%** = excellent, **<20%** = good, **>30%** = poor")
        lines.append("")
        lines.append("### Character Error Rate (CER)")
        lines.append("- Character-level equivalent of WER.")
        lines.append("- Useful for languages without clear word boundaries or fine-grained analysis.")
        lines.append("")
        lines.append("### Intent Accuracy")
        lines.append("- Fraction of correctly classified intents.")
        lines.append("- **>90%** = excellent for a rule-based system.")
        lines.append("")

    return "\n".join(lines)
