from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def compare_baseline_and_enhanced(
    baseline_predictions,
    enhanced_predictions,
):
    baseline_f1 = f1_score(
        baseline_predictions["target"],
        baseline_predictions["prediction"],
    )

    enhanced_f1 = f1_score(
        enhanced_predictions["target"],
        enhanced_predictions["prediction"],
    )

    baseline_auc = roc_auc_score(
        baseline_predictions["target"],
        baseline_predictions["probability"],
    )

    enhanced_auc = roc_auc_score(
        enhanced_predictions["target"],
        enhanced_predictions["probability"],
    )

    f1_relative_improvement = (
        (enhanced_f1 - baseline_f1) / baseline_f1 * 100
        if baseline_f1 else 0
    )

    return {
        "baseline_f1": float(baseline_f1),
        "enhanced_f1": float(enhanced_f1),
        "f1_relative_improvement_pct": float(
            f1_relative_improvement
        ),
        "baseline_roc_auc": float(baseline_auc),
        "enhanced_roc_auc": float(enhanced_auc),
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
