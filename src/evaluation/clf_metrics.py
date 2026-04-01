"""
分类评估 + 时间成本
"""
import json
import os
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

from ..utils.config import RESULTS_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ClassificationEvaluator:

    def __init__(self, label_map: Dict = None):
        self.label_map = label_map  # {id: name}
        self.metrics = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 time_costs: Dict = None) -> Dict:
        """
        计算分类指标 + 可选时间成本

        time_costs 可包含: feature_construction_time, train_time,
            inference_time_total, inference_time_per_sample
        """
        self.metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        # per-class precision / recall / f1
        labels = sorted(set(y_true) | set(y_pred))
        per_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        per_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        per_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

        per_class = {}
        for i, lbl in enumerate(labels):
            name = self.label_map.get(lbl, f"class_{lbl}") if self.label_map else f"class_{lbl}"
            per_class[name] = {
                "precision": float(per_prec[i]),
                "recall": float(per_rec[i]),
                "f1": float(per_f1[i]),
            }
        self.metrics["per_class"] = per_class

        self.metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        if time_costs:
            self.metrics["time_costs"] = time_costs

        logger.info("Accuracy=%.4f  Macro-F1=%.4f  Weighted-F1=%.4f",
                     self.metrics["accuracy"],
                     self.metrics["macro_f1"],
                     self.metrics["weighted_f1"])
        return self.metrics

    def save_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        logger.info("Metrics JSON saved: %s", filepath)

    def save_per_class_csv(self, filepath: str):
        """保存 per-class 表到 CSV"""
        if "per_class" not in self.metrics:
            return
        rows = []
        for name, vals in self.metrics["per_class"].items():
            rows.append({"device": name, **vals})
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info("Per-class CSV saved: %s", filepath)

    def save_confusion_matrix_plot(self, filepath: str, labels=None):
        """保存混淆矩阵热力图"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(self.metrics["confusion_matrix"])
        if labels is None and self.label_map:
            n = cm.shape[0]
            labels = [self.label_map.get(i, str(i)) for i in range(n)]

        fig, ax = plt.subplots(figsize=(max(8, len(cm) * 0.6), max(6, len(cm) * 0.5)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix plot saved: %s", filepath)

    def get_report(self, y_true, y_pred) -> str:
        target_names = None
        if self.label_map:
            unique_labels = sorted(set(y_true) | set(y_pred))
            target_names = [self.label_map.get(l, f"class_{l}") for l in unique_labels]
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def compute_metrics(y_true, y_pred, label_map=None, time_costs=None,
                    save_dir=None, tag="") -> Dict:
    """一站式评估并保存"""
    evaluator = ClassificationEvaluator(label_map=label_map)
    metrics = evaluator.evaluate(y_true, y_pred, time_costs=time_costs)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        evaluator.save_json(os.path.join(save_dir, f"{tag}_metrics.json"))
        evaluator.save_per_class_csv(os.path.join(save_dir, f"{tag}_per_class.csv"))
        evaluator.save_confusion_matrix_plot(os.path.join(save_dir, f"{tag}_confusion.png"))

    return metrics
