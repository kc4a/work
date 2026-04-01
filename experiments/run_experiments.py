"""
实验组调度器
Group A: 单包级 + ML
Group B: 窗口级 + ML
Group C: 序列压缩 + ML
Group D: 原始序列 + RNN
+ 参数敏感性分析 + 特征重要性分析
"""
import os
import json
import time
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import (
    RESULTS_DIR, RANDOM_STATE,
    WINDOW_SIZES, WINDOW_STEPS, DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STEP,
    SEQ_LENGTHS, SEQ_STEPS, DEFAULT_SEQ_LENGTH, DEFAULT_SEQ_STEP,
    RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, RNN_DROPOUT, RNN_BATCH_SIZE,
    RNN_LEARNING_RATE, RNN_EPOCHS, RNN_PATIENCE, DEVICE,
)
from src.preprocessing.pcap_loader import parse_pcap_to_packet_csv
from src.preprocessing.packet_level_loader import GroupALoader
from src.preprocessing.window_loader import GroupBLoader
from src.preprocessing.sequence_loader import GroupCLoader, GroupDLoader
from src.models.classifier import IoTClassifier
from src.models.rnn_classifier import (
    IoTRNNClassifier, RNNTrainer, create_dataloaders,
)
from src.evaluation.clf_metrics import compute_metrics

logger = logging.getLogger(__name__)

EXPR_DIR = os.path.join(RESULTS_DIR, "experiments")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


# ═══════════════════════════════════════════════════
# Group A
# ═══════════════════════════════════════════════════

def run_group_a(model_type="rf"):
    tag = f"group_a_{model_type}"
    logger.info("=== %s ===", tag)

    t_feat = time.time()
    loader = GroupALoader()
    X_train, y_train, X_val, y_val, X_test, y_test, info = loader.load()
    feat_time = time.time() - t_feat

    clf = IoTClassifier(model_type)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)

    time_costs = {
        "feature_construction_time": feat_time,
        "train_time": clf.train_time,
        "inference_time_total": clf.inference_time_total,
        "inference_time_per_sample": clf.inference_time_per_sample,
    }

    inv_map = {v: k for k, v in info["label_map"].items()}
    metrics = compute_metrics(
        y_test, y_pred, label_map=inv_map, time_costs=time_costs,
        save_dir=os.path.join(EXPR_DIR, tag), tag=tag,
    )
    metrics["group"] = "A"
    metrics["model"] = model_type
    metrics["info"] = {k: v for k, v in info.items() if k != "feature_names"}
    _save_result_row(tag, metrics)
    return metrics


# ═══════════════════════════════════════════════════
# Group B
# ═══════════════════════════════════════════════════

def run_group_b(model_type="rf", window_size=DEFAULT_WINDOW_SIZE,
                step=DEFAULT_WINDOW_STEP):
    tag = f"group_b_{model_type}_W{window_size}_S{step}"
    logger.info("=== %s ===", tag)

    loader = GroupBLoader()
    X_train, y_train, X_val, y_val, X_test, y_test, info = loader.load(
        window_size=window_size, step=step,
    )
    feat_time = info["feature_construction_time"]

    clf = IoTClassifier(model_type)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)

    time_costs = {
        "feature_construction_time": feat_time,
        "train_time": clf.train_time,
        "inference_time_total": clf.inference_time_total,
        "inference_time_per_sample": clf.inference_time_per_sample,
    }

    inv_map = {v: k for k, v in info["label_map"].items()}
    metrics = compute_metrics(
        y_test, y_pred, label_map=inv_map, time_costs=time_costs,
        save_dir=os.path.join(EXPR_DIR, tag), tag=tag,
    )
    metrics["group"] = "B"
    metrics["model"] = model_type
    metrics["window_size"] = window_size
    metrics["window_step"] = step
    metrics["info"] = {k: v for k, v in info.items() if k != "feature_names"}
    _save_result_row(tag, metrics)
    return metrics


# ═══════════════════════════════════════════════════
# Group C
# ═══════════════════════════════════════════════════

def run_group_c(model_type="rf", seq_length=DEFAULT_SEQ_LENGTH,
                step=DEFAULT_SEQ_STEP):
    tag = f"group_c_{model_type}_H{seq_length}_L{step}"
    logger.info("=== %s ===", tag)

    loader = GroupCLoader()
    X_train, y_train, X_val, y_val, X_test, y_test, info = loader.load(
        seq_length=seq_length, step=step,
    )
    feat_time = info["feature_construction_time"]

    clf = IoTClassifier(model_type)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)

    time_costs = {
        "feature_construction_time": feat_time,
        "train_time": clf.train_time,
        "inference_time_total": clf.inference_time_total,
        "inference_time_per_sample": clf.inference_time_per_sample,
    }

    inv_map = {v: k for k, v in info["label_map"].items()}
    metrics = compute_metrics(
        y_test, y_pred, label_map=inv_map, time_costs=time_costs,
        save_dir=os.path.join(EXPR_DIR, tag), tag=tag,
    )
    metrics["group"] = "C"
    metrics["model"] = model_type
    metrics["seq_length"] = seq_length
    metrics["seq_step"] = step
    metrics["info"] = {k: v for k, v in info.items() if k != "feature_names"}
    _save_result_row(tag, metrics)
    return metrics


# ═══════════════════════════════════════════════════
# Group D
# ═══════════════════════════════════════════════════

def run_group_d(rnn_type="lstm", seq_length=DEFAULT_SEQ_LENGTH,
                step=DEFAULT_SEQ_STEP):
    tag = f"group_d_{rnn_type}_H{seq_length}_L{step}"
    logger.info("=== %s ===", tag)

    loader = GroupDLoader()
    X_train, y_train, X_val, y_val, X_test, y_test, info = loader.load(
        seq_length=seq_length, step=step,
    )
    feat_time = info["feature_construction_time"]

    n_feat = info["n_features_per_step"]
    n_classes = info["n_classes"]

    model = IoTRNNClassifier(
        input_size=n_feat,
        hidden_size=RNN_HIDDEN_SIZE,
        num_layers=RNN_NUM_LAYERS,
        num_classes=n_classes,
        rnn_type=rnn_type,
        dropout=RNN_DROPOUT,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=RNN_BATCH_SIZE,
    )

    trainer = RNNTrainer(model, device=DEVICE, lr=RNN_LEARNING_RATE)
    save_path = os.path.join(RESULTS_DIR, "models", f"{tag}_best.pt")
    history = trainer.train(
        train_loader, val_loader,
        epochs=RNN_EPOCHS, patience=RNN_PATIENCE,
        save_path=save_path,
    )

    t_inf_start = time.perf_counter()
    _, test_acc, y_true, y_pred = trainer.evaluate(test_loader, measure_latency=True)
    t_inf_end = time.perf_counter()

    inference_total = t_inf_end - t_inf_start
    inference_per_sample = inference_total / max(len(y_true), 1)

    time_costs = {
        "feature_construction_time": feat_time,
        "train_time": history["train_time"],
        "inference_time_total": inference_total,
        "inference_time_per_sample": inference_per_sample,
    }

    inv_map = {v: k for k, v in info["label_map"].items()}
    metrics = compute_metrics(
        y_true, y_pred, label_map=inv_map, time_costs=time_costs,
        save_dir=os.path.join(EXPR_DIR, tag), tag=tag,
    )
    metrics["group"] = "D"
    metrics["model"] = rnn_type
    metrics["seq_length"] = seq_length
    metrics["seq_step"] = step
    metrics["info"] = {k: v for k, v in info.items()
                        if k not in ("feature_names",)}
    _save_result_row(tag, metrics)
    return metrics


# ═══════════════════════════════════════════════════
# 参数敏感性分析
# ═══════════════════════════════════════════════════

def run_parameter_sensitivity_window():
    """窗口参数敏感性: 先变 W（固定 S=W），再变 S（固定最佳 W*）"""
    logger.info("=== 窗口参数敏感性分析 ===")

    results_a = []
    for W in WINDOW_SIZES:
        for mt in ("rf", "xgb"):
            m = run_group_b(mt, window_size=W, step=W)
            results_a.append({
                "W": W, "S": W, "model": mt,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                **m.get("time_costs", {}),
            })

    df_a = pd.DataFrame(results_a)
    df_a.to_csv(os.path.join(TABLE_DIR, "sensitivity_window_W.csv"), index=False)

    best_W = int(df_a.groupby("W")["macro_f1"].mean().idxmax())
    logger.info("最佳窗口大小 W*=%d", best_W)

    results_b = []
    for S in WINDOW_STEPS:
        if S > best_W:
            continue
        for mt in ("rf", "xgb"):
            m = run_group_b(mt, window_size=best_W, step=S)
            results_b.append({
                "W": best_W, "S": S, "model": mt,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                **m.get("time_costs", {}),
            })

    if results_b:
        df_b = pd.DataFrame(results_b)
        df_b.to_csv(os.path.join(TABLE_DIR, "sensitivity_window_S.csv"), index=False)

    _plot_sensitivity(df_a, "W", "Window Size (s)",
                      os.path.join(FIG_DIR, "sensitivity_window_W.png"))


def run_parameter_sensitivity_sequence():
    """序列参数敏感性: 先变 H（固定 L=H），再变 L（固定最佳 H*）"""
    logger.info("=== 序列参数敏感性分析 ===")

    results_a = []
    for H in SEQ_LENGTHS:
        for mt in ("rf", "xgb"):
            m = run_group_c(mt, seq_length=H, step=H)
            results_a.append({
                "H": H, "L": H, "model": mt, "group": "C",
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                **m.get("time_costs", {}),
            })
        for rt in ("lstm", "gru"):
            m = run_group_d(rt, seq_length=H, step=H)
            results_a.append({
                "H": H, "L": H, "model": rt, "group": "D",
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                **m.get("time_costs", {}),
            })

    df_a = pd.DataFrame(results_a)
    df_a.to_csv(os.path.join(TABLE_DIR, "sensitivity_seq_H.csv"), index=False)

    best_H = int(df_a.groupby("H")["macro_f1"].mean().idxmax())
    logger.info("最佳序列长度 H*=%d", best_H)

    results_b = []
    for L in SEQ_STEPS:
        if L > best_H:
            continue
        for mt in ("rf", "xgb"):
            m = run_group_c(mt, seq_length=best_H, step=L)
            results_b.append({
                "H": best_H, "L": L, "model": mt, "group": "C",
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                **m.get("time_costs", {}),
            })
        for rt in ("lstm", "gru"):
            m = run_group_d(rt, seq_length=best_H, step=L)
            results_b.append({
                "H": best_H, "L": L, "model": rt, "group": "D",
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                **m.get("time_costs", {}),
            })

    if results_b:
        df_b = pd.DataFrame(results_b)
        df_b.to_csv(os.path.join(TABLE_DIR, "sensitivity_seq_L.csv"), index=False)

    _plot_sensitivity(df_a, "H", "Sequence Length (packets)",
                      os.path.join(FIG_DIR, "sensitivity_seq_H.png"))


# ═══════════════════════════════════════════════════
# 特征重要性分析
# ═══════════════════════════════════════════════════

def run_feature_importance():
    """对 Group A / B / C 的 RF 模型做特征重要性分析"""
    logger.info("=== 特征重要性分析 ===")

    groups = [
        ("A", GroupALoader, {}),
        ("B", GroupBLoader, {"window_size": DEFAULT_WINDOW_SIZE, "step": DEFAULT_WINDOW_STEP}),
        ("C", GroupCLoader, {"seq_length": DEFAULT_SEQ_LENGTH, "step": DEFAULT_SEQ_STEP}),
    ]

    for gname, LoaderCls, kwargs in groups:
        loader = LoaderCls()
        X_train, y_train, X_val, y_val, X_test, y_test, info = loader.load(**kwargs)

        clf = IoTClassifier("rf")
        clf.train(X_train, y_train)

        importances = clf.get_feature_importance()
        if importances is None:
            continue

        feat_names = info["feature_names"]
        imp_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        csv_path = os.path.join(TABLE_DIR, f"feature_importance_group_{gname}.csv")
        imp_df.to_csv(csv_path, index=False)

        _plot_feature_importance(imp_df, gname,
                                 os.path.join(FIG_DIR, f"feature_importance_group_{gname}.png"))

        logger.info("Group %s top-5 features: %s", gname,
                     imp_df.head(5)[["feature", "importance"]].to_dict("records"))


# ═══════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════

def aggregate_results():
    """汇总所有实验的总体性能表"""
    summary_path = os.path.join(EXPR_DIR, "all_results.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        logger.info("总体性能表 (%d 条记录):\n%s", len(df), df.to_string(index=False))
        df.to_csv(os.path.join(TABLE_DIR, "overall_summary.csv"), index=False)
    else:
        logger.warning("未找到 all_results.csv")


# ═══════════════════════════════════════════════════
# 完整流水线
# ═══════════════════════════════════════════════════

def run_all():
    """按实验计划的推荐执行顺序运行"""

    # 阶段 1: 生成 packet-level CSV
    logger.info("===== 阶段 1: 生成 packet-level CSV =====")
    parse_pcap_to_packet_csv()

    # 阶段 2: 四组主实验（默认参数）
    logger.info("===== 阶段 2: 四组主实验 =====")
    for mt in ("rf", "xgb"):
        run_group_a(mt)
    for mt in ("rf", "xgb"):
        run_group_b(mt)
    for mt in ("rf", "xgb"):
        run_group_c(mt)
    for rt in ("lstm", "gru"):
        run_group_d(rt)

    # 阶段 3: 参数敏感性
    logger.info("===== 阶段 3: 参数敏感性分析 =====")
    run_parameter_sensitivity_window()
    run_parameter_sensitivity_sequence()

    # 阶段 4: 特征重要性
    logger.info("===== 阶段 4: 特征重要性分析 =====")
    run_feature_importance()

    # 阶段 5: 汇总
    logger.info("===== 阶段 5: 汇总结果 =====")
    aggregate_results()

    logger.info("===== 全部实验完成 =====")


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def _save_result_row(tag, metrics):
    """追加一行到 all_results.csv"""
    row = {
        "tag": tag,
        "group": metrics.get("group", ""),
        "model": metrics.get("model", ""),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
    }
    tc = metrics.get("time_costs", {})
    row.update({
        "feature_construction_time": tc.get("feature_construction_time"),
        "train_time": tc.get("train_time"),
        "inference_time_total": tc.get("inference_time_total"),
        "inference_time_per_sample": tc.get("inference_time_per_sample"),
    })
    for key in ("window_size", "window_step", "seq_length", "seq_step"):
        if key in metrics:
            row[key] = metrics[key]

    csv_path = os.path.join(EXPR_DIR, "all_results.csv")
    df_new = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


def _plot_sensitivity(df, x_col, x_label, filepath):
    """绘制参数敏感性图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for metric, ax in zip(["accuracy", "macro_f1", "weighted_f1"], axes):
        for model_name, mdf in df.groupby("model"):
            ax.plot(mdf[x_col], mdf[metric], marker="o", label=model_name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Sensitivity plot saved: %s", filepath)


def _plot_feature_importance(imp_df, group_name, filepath, top_n=20):
    """绘制特征重要性柱状图"""
    top = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(range(len(top)), top["importance"].values, align="center")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Group {group_name} — Top {top_n} Feature Importances (RF)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved: %s", filepath)
