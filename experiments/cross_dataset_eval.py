"""
跨数据集泛化能力评估
在原始数据集上训练模型，在新数据集上测试。
"""
import os
import time
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

from src.utils.config import (
    RESULTS_DIR, PROCESSED_DIR, BASE_DIR,
    GROUP_A_FEATURES, GROUP_D_FEATURES, DEVICE,
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STEP,
    DEFAULT_GROUP_E_SAMPLE_STEP, DEFAULT_GROUP_E_HIST_WINDOW,
    DEFAULT_GROUP_E1_D_MAX,
)
from src.preprocessing.pcap_loader import parse_pcap_to_packet_csv, load_packet_csv
from src.preprocessing.window_loader import GroupBLoader
from src.preprocessing.group_e_loader import GroupELoader
from src.preprocessing.sequence_loader import GroupCLoader, GroupDLoader, _build_sequences
from src.preprocessing.group_e1_loader import (
    GroupE1Loader, CUR_PACKET_FIELDS, HIST_BASELINE_KEYS,
    _compute_relation_features,
)
from src.features.sequence_compressed_features import compute_sequence_compressed_features
from src.features.window_features import compute_window_features
from src.models.classifier import IoTClassifier
from src.models.rnn_classifier import IoTRNNClassifier, RNNTrainer, create_dataloaders
from src.evaluation.clf_metrics import compute_metrics

logger = logging.getLogger(__name__)

EXPR_DIR = os.path.join(RESULTS_DIR, "experiments")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

# ── 新数据集配置 ─────────────────────────────────────
NEW_PCAP_PATH = os.path.join(BASE_DIR, "data", "new_data", "IoT_2023-07-24.pcap")
NEW_CSV_PATH = os.path.join(PROCESSED_DIR, "new_packet_level.csv")
CROSS_GROUP_C_CONFIG = {
    "rf": {"seq_length": 50, "step": 10},
    "xgb": {"seq_length": 50, "step": 25},
}
CROSS_GROUP_D_CONFIG = {
    "lstm": {"seq_length": 50, "step": 10},
    "gru": {"seq_length": 50, "step": 10},
}

# 新 MAC → 旧训练标签映射
NEW_MAC_TO_DEVICE = {
    "40:f6:bc:bc:89:7b": "Amazon Echo",             # Echo Dot (4th Gen)
    "18:48:be:31:4b:49": "Amazon Echo",             # Echo Show 8
    "74:d4:23:32:a2:d7": "Amazon Echo",             # Echo Show 8
    "70:ee:50:57:95:29": "Netatmo Welcome",         # Netatmo Smart Indoor Security Camera
    "54:af:97:bb:8d:8f": "TP-Link Day Night Cloud camera",  # TP-Link Tapo
    "00:16:6c:d7:d5:f9": "Samsung SmartCam",        # SAMSUNG Pan/Tilt Camera
    "70:ee:50:96:bb:dc": "Netatmo weather station",  # Netatmo Weather Station
    "b0:02:47:6f:63:37": "PIX-STAR Photo-frame",    # Pix-Star Digital Photo Frame
    "84:69:93:27:ad:35": "HP Printer",              # HP Envy
    "68:3a:48:0d:d4:1c": "Smart Things",            # Aeotec Smart Hub
}


def _build_classifier(model_type):
    """Use single-threaded tree learners to avoid Windows sandbox issues."""
    return IoTClassifier(model_type, n_jobs=1)


def _full_label_encoder():
    """Fit labels on the full original dataset to match saved C/D models."""
    old_df = load_packet_csv()
    le = LabelEncoder()
    le.fit(old_df["device_id"].unique())
    return old_df, le


def _build_group_c_features_for_devices(df, seq_length, step, split_filter, devices):
    subset = df[df["device_id"].isin(devices)]
    if split_filter:
        subset = subset[subset["split"] == split_filter]

    rows = []
    labels = []
    if split_filter is None:
        for device_id, dev_df in subset.groupby("device_id"):
            dev_df = dev_df.sort_values("ts").reset_index(drop=True)
            n = len(dev_df)
            start = 0
            while start + seq_length <= n:
                seq_df = dev_df.iloc[start:start + seq_length]
                rows.append(compute_sequence_compressed_features(seq_df))
                labels.append(device_id)
                start += step
    else:
        seqs, seq_labels = _build_sequences(subset, seq_length, step, split_filter)
        for seq_df, device_id in zip(seqs, seq_labels):
            rows.append(compute_sequence_compressed_features(seq_df))
            labels.append(device_id)

    feat_df = pd.DataFrame(rows).fillna(0)
    return feat_df, labels


def _ensure_group_c_model(model_type, seq_length, step):
    model_path = os.path.join(
        RESULTS_DIR, "models", f"group_c_{model_type}_H{seq_length}_L{step}.pkl"
    )
    if os.path.exists(model_path):
        clf = _build_classifier(model_type)
        clf.load(model_path)
        return clf, model_path

    logger.info("未找到 Group C 模型，开始训练并保存: %s", model_path)
    loader = GroupCLoader()
    X_train, y_train, X_val, y_val, X_test, y_test, info = loader.load(
        seq_length=seq_length, step=step
    )
    clf = _build_classifier(model_type)
    clf.train(X_train, y_train)
    clf.save(model_path)
    return clf, model_path


def parse_new_pcap(force=False):
    """解析新数据集 PCAP，使用映射后的设备名，全部标记为 test"""
    return parse_pcap_to_packet_csv(
        pcap_path=NEW_PCAP_PATH,
        output_path=NEW_CSV_PATH,
        force=force,
        mac_to_device=NEW_MAC_TO_DEVICE,
        min_samples=0,
        test_only=True,
    )


# ═══════════════════════════════════════════════════
# Group A 跨数据集
# ═══════════════════════════════════════════════════

def cross_dataset_group_a(model_type="rf"):
    tag = f"cross_dataset_group_a_{model_type}"
    logger.info("=== %s ===", tag)

    # 加载原始训练数据
    old_df = load_packet_csv()
    new_df = load_packet_csv(NEW_CSV_PATH)

    # 共同设备
    common_devices = sorted(set(old_df["device_id"].unique()) & set(new_df["device_id"].unique()))
    logger.info("共同设备 (%d): %s", len(common_devices), common_devices)

    if not common_devices:
        logger.error("无共同设备，跳过")
        return None

    # 标签编码（仅共同设备）
    le = LabelEncoder()
    le.fit(common_devices)
    label_map = {name: idx for idx, name in enumerate(le.classes_)}

    # 训练集（原始数据，仅共同设备，仅 train split）
    old_train = old_df[(old_df["split"] == "train") & (old_df["device_id"].isin(common_devices))].copy()
    # 测试集（新数据，仅共同设备）
    new_test = new_df[new_df["device_id"].isin(common_devices)].copy()

    features = GROUP_A_FEATURES
    X_train = old_train[features].fillna(0).values.astype(np.float32)
    y_train = le.transform(old_train["device_id"])
    X_test = new_test[features].fillna(0).values.astype(np.float32)
    y_test = le.transform(new_test["device_id"])

    logger.info("训练: %d 样本, 测试: %d 样本, %d 类", len(X_train), len(X_test), len(common_devices))

    # 训练 & 预测
    t0 = time.time()
    clf = _build_classifier(model_type)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)
    total_time = time.time() - t0

    time_costs = {
        "train_time": clf.train_time,
        "inference_time_total": clf.inference_time_total,
        "inference_time_per_sample": clf.inference_time_per_sample,
    }

    inv_map = {v: k for k, v in label_map.items()}
    save_dir = os.path.join(EXPR_DIR, tag)
    metrics = compute_metrics(y_test, y_pred, label_map=inv_map,
                              time_costs=time_costs, save_dir=save_dir, tag=tag)
    return metrics


# ═══════════════════════════════════════════════════
# Group B 跨数据集
# ═══════════════════════════════════════════════════

def cross_dataset_group_b(model_type="rf",
                          window_size=DEFAULT_WINDOW_SIZE,
                          step=DEFAULT_WINDOW_STEP):
    tag = f"cross_dataset_group_b_{model_type}_W{window_size}_S{step}"
    logger.info("=== %s ===", tag)

    old_df = load_packet_csv()
    new_df = load_packet_csv(NEW_CSV_PATH)

    common_devices = sorted(set(old_df["device_id"].unique()) & set(new_df["device_id"].unique()))
    logger.info("共同设备 (%d): %s", len(common_devices), common_devices)

    if not common_devices:
        logger.error("无共同设备，跳过")
        return None

    le = LabelEncoder()
    le.fit(common_devices)

    # 构造窗口特征
    def _build_windows(df, split_filter, devices):
        from src.utils.config import MIN_PACKETS_PER_WINDOW
        subset = df[df["device_id"].isin(devices)]
        if split_filter:
            subset = subset[subset["split"] == split_filter]
        rows = []
        for device_id, dev_df in subset.groupby("device_id"):
            dev_df = dev_df.sort_values("ts").reset_index(drop=True)
            ts_arr = dev_df["ts"].values
            if len(ts_arr) == 0:
                continue
            t_start, t_end = ts_arr[0], ts_arr[-1]
            win_start = t_start
            while win_start < t_end:
                win_end = win_start + window_size
                mask = (ts_arr >= win_start) & (ts_arr < win_end)
                win_df = dev_df.loc[mask]
                if len(win_df) >= MIN_PACKETS_PER_WINDOW:
                    feat = compute_window_features(win_df)
                    feat["device_id"] = device_id
                    rows.append(feat)
                win_start += step
        return pd.DataFrame(rows)

    train_feat = _build_windows(old_df, "train", common_devices)
    test_feat = _build_windows(new_df, None, common_devices)

    if train_feat.empty or test_feat.empty:
        logger.error("特征构造结果为空")
        return None

    meta_cols = {"device_id"}
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train = train_feat[feature_cols].fillna(0).values.astype(np.float32)
    y_train = le.transform(train_feat["device_id"])
    X_test = test_feat[feature_cols].fillna(0).values.astype(np.float32)
    y_test = le.transform(test_feat["device_id"])

    logger.info("训练: %d 样本, 测试: %d 样本", len(X_train), len(X_test))

    clf = _build_classifier(model_type)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)

    time_costs = {
        "train_time": clf.train_time,
        "inference_time_total": clf.inference_time_total,
        "inference_time_per_sample": clf.inference_time_per_sample,
    }

    inv_map = {v: k for k, v in dict(enumerate(le.classes_)).items()}
    save_dir = os.path.join(EXPR_DIR, tag)
    metrics = compute_metrics(y_test, y_pred, label_map=inv_map,
                              time_costs=time_costs, save_dir=save_dir, tag=tag)
    return metrics


def cross_dataset_group_c(model_type="rf", seq_length=None, step=None):
    cfg = CROSS_GROUP_C_CONFIG[model_type]
    seq_length = seq_length or cfg["seq_length"]
    step = step or cfg["step"]
    tag = f"cross_dataset_group_c_{model_type}_H{seq_length}_L{step}"
    logger.info("=== %s ===", tag)

    old_df, le = _full_label_encoder()
    new_df = load_packet_csv(NEW_CSV_PATH)

    common_devices = sorted(
        set(old_df["device_id"].unique()) & set(new_df["device_id"].unique())
    )
    logger.info("共同设备 (%d): %s", len(common_devices), common_devices)

    if not common_devices:
        logger.error("无共同设备，跳过")
        return None

    clf, model_path = _ensure_group_c_model(model_type, seq_length, step)
    logger.info("使用 Group C 模型: %s", model_path)

    test_feat, test_labels = _build_group_c_features_for_devices(
        new_df, seq_length, step, None, common_devices
    )
    if test_feat.empty:
        logger.error("Group C 测试特征为空")
        return None

    feature_cols = list(test_feat.columns)
    X_test = test_feat[feature_cols].fillna(0).values.astype(np.float32)
    y_test = le.transform(test_labels)
    y_pred = clf.predict(X_test)

    time_costs = {
        "inference_time_total": clf.inference_time_total,
        "inference_time_per_sample": clf.inference_time_per_sample,
    }

    inv_map = {idx: name for idx, name in enumerate(le.classes_)}
    save_dir = os.path.join(EXPR_DIR, tag)
    metrics = compute_metrics(
        y_test, y_pred, label_map=inv_map, time_costs=time_costs,
        save_dir=save_dir, tag=tag,
    )
    return metrics


# ═══════════════════════════════════════════════════
# Group E 跨数据集
# ═══════════════════════════════════════════════════

def cross_dataset_group_e(model_type="rf",
                          sample_step=DEFAULT_GROUP_E_SAMPLE_STEP,
                          hist_window=DEFAULT_GROUP_E_HIST_WINDOW):
    tag = f"cross_dataset_group_e_{model_type}_S{sample_step}_W{hist_window}"
    logger.info("=== %s ===", tag)

    old_df = load_packet_csv()
    new_df = load_packet_csv(NEW_CSV_PATH)

    common_devices = sorted(set(old_df["device_id"].unique()) & set(new_df["device_id"].unique()))
    logger.info("共同设备 (%d): %s", len(common_devices), common_devices)

    if not common_devices:
        logger.error("无共同设备，跳过")
        return None

    le = LabelEncoder()
    le.fit(common_devices)

    from src.preprocessing.group_e_loader import _init_window_feature_keys, _get_empty_window_features

    # 初始化窗口特征 key
    _init_window_feature_keys(old_df.head(10))

    def _build_group_e_samples(df, split_filter, devices):
        subset = df[df["device_id"].isin(devices)]
        if split_filter:
            subset = subset[subset["split"] == split_filter]
        rows = []
        for device_id, dev_df in subset.groupby("device_id"):
            dev_df = dev_df.sort_values("ts").reset_index(drop=True)
            ts_arr = dev_df["ts"].values
            if len(ts_arr) < 2:
                continue
            t_start, t_end = ts_arr[0], ts_arr[-1]
            anchor = t_start + sample_step
            while anchor <= t_end:
                idx = np.searchsorted(ts_arr, anchor, side="left")
                if idx >= len(ts_arr):
                    break
                t_cur = ts_arr[idx]
                cur_pkt = dev_df.iloc[idx]

                pkt_feat = {f"pkt_{f}": cur_pkt.get(f, 0.0) for f in GROUP_A_FEATURES}

                hist_start = t_cur - hist_window
                hist_mask = (ts_arr >= hist_start) & (ts_arr < t_cur)
                hist_df = dev_df.loc[hist_mask]
                n_hist = len(hist_df)

                if n_hist >= 1:
                    win_feat = compute_window_features(hist_df)
                else:
                    win_feat = _get_empty_window_features()

                hist_feat = {f"hist_{k}": v for k, v in win_feat.items()}

                if n_hist >= 2:
                    actual_dur = float(ts_arr[hist_mask][-1] - ts_arr[hist_mask][0])
                else:
                    actual_dur = 0.0

                total_bytes = float(hist_df["packet_len"].sum()) if n_hist > 0 else 0.0
                safe_dur = max(actual_dur, 1e-6)

                cold_feat = {
                    "hist_duration_actual": actual_dur,
                    "hist_window_ratio": actual_dur / hist_window,
                    "hist_complete_flag": 1.0 if actual_dur >= hist_window * 0.9 else 0.0,
                    "hist_packet_count_actual": float(n_hist),
                    "packet_rate": n_hist / safe_dur,
                    "byte_rate": total_bytes / safe_dur,
                }

                row = {}
                row.update(pkt_feat)
                row.update(hist_feat)
                row.update(cold_feat)
                row["device_id"] = device_id
                rows.append(row)
                anchor += sample_step
        return pd.DataFrame(rows)

    logger.info("构造训练集 Group E 特征...")
    train_feat = _build_group_e_samples(old_df, "train", common_devices)
    logger.info("构造测试集 Group E 特征...")
    test_feat = _build_group_e_samples(new_df, None, common_devices)

    if train_feat.empty or test_feat.empty:
        logger.error("特征构造结果为空")
        return None

    meta_cols = {"device_id"}
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    X_train = train_feat[feature_cols].fillna(0).values.astype(np.float32)
    y_train = le.transform(train_feat["device_id"])
    X_test = test_feat[feature_cols].fillna(0).values.astype(np.float32)
    y_test = le.transform(test_feat["device_id"])

    logger.info("训练: %d 样本, 测试: %d 样本", len(X_train), len(X_test))

    clf = _build_classifier(model_type)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)

    time_costs = {
        "train_time": clf.train_time,
        "inference_time_total": clf.inference_time_total,
        "inference_time_per_sample": clf.inference_time_per_sample,
    }

    inv_map = {v: k for k, v in dict(enumerate(le.classes_)).items()}
    save_dir = os.path.join(EXPR_DIR, tag)
    metrics = compute_metrics(y_test, y_pred, label_map=inv_map,
                              time_costs=time_costs, save_dir=save_dir, tag=tag)
    return metrics


def cross_dataset_group_e1(model_type="rf",
                           sample_step=DEFAULT_GROUP_E_SAMPLE_STEP,
                           hist_window=DEFAULT_GROUP_E_HIST_WINDOW,
                           d_max=DEFAULT_GROUP_E1_D_MAX):
    tag = f"cross_dataset_group_e1_{model_type}_S{sample_step}_W{hist_window}"
    logger.info("=== %s ===", tag)

    old_df = load_packet_csv()
    new_df = load_packet_csv(NEW_CSV_PATH)

    common_devices = sorted(
        set(old_df["device_id"].unique()) & set(new_df["device_id"].unique())
    )
    logger.info("共同设备 (%d): %s", len(common_devices), common_devices)

    if not common_devices:
        logger.error("无共同设备，跳过")
        return None

    le = LabelEncoder()
    le.fit(common_devices)

    def _build_group_e1_samples(df, split_filter, devices):
        subset = df[df["device_id"].isin(devices)]
        if split_filter:
            subset = subset[subset["split"] == split_filter]
        rows = []
        for device_id, dev_df in subset.groupby("device_id"):
            dev_df = dev_df.sort_values("ts").reset_index(drop=True)
            ts_arr = dev_df["ts"].values
            payload_arr = dev_df["payload_len"].values
            if len(ts_arr) < 2:
                continue
            t_start, t_end = ts_arr[0], ts_arr[-1]
            anchor = t_start + sample_step
            while anchor <= t_end:
                cur_idx = GroupE1Loader._select_packet(ts_arr, payload_arr, anchor, d_max)
                if cur_idx is None:
                    anchor += sample_step
                    continue

                t_cur = ts_arr[cur_idx]
                cur_pkt = dev_df.iloc[cur_idx]
                anchor_gap = anchor - t_cur

                cur_feat = {
                    f"cur_{f}": float(cur_pkt.get(f, 0.0))
                    for f in CUR_PACKET_FIELDS
                }

                hist_start = t_cur - hist_window
                hist_mask = (ts_arr >= hist_start) & (ts_arr < t_cur)
                hist_df = dev_df.loc[hist_mask]
                n_hist = len(hist_df)

                if n_hist >= 1:
                    win_stats = compute_window_features(hist_df)
                else:
                    win_stats = {k: 0.0 for k in HIST_BASELINE_KEYS}

                hist_feat = {
                    f"hist_{k}": float(win_stats.get(k, 0.0))
                    for k in HIST_BASELINE_KEYS
                }

                if n_hist >= 2:
                    actual_dur = float(ts_arr[hist_mask][-1] - ts_arr[hist_mask][0])
                else:
                    actual_dur = 0.0
                safe_dur = max(actual_dur, 1e-6)
                hist_feat["hist_packet_rate"] = n_hist / safe_dur
                hist_feat["hist_byte_rate"] = float(
                    win_stats.get("total_bytes", 0.0)
                ) / safe_dur

                rel_feat = _compute_relation_features(cur_pkt, win_stats, hist_df)

                ratio = actual_dur / hist_window if hist_window > 0 else 0.0
                if ratio < 0.3:
                    stage = 0
                elif ratio < 0.8:
                    stage = 1
                else:
                    stage = 2

                reliability_feat = {
                    "hist_duration_actual": actual_dur,
                    "hist_window_ratio": ratio,
                    "hist_packet_count_actual": float(n_hist),
                    "hist_stage": float(stage),
                    "anchor_gap": anchor_gap,
                }

                row = {}
                row.update(cur_feat)
                row.update(hist_feat)
                row.update(rel_feat)
                row.update(reliability_feat)
                row["device_id"] = device_id
                rows.append(row)
                anchor += sample_step
        return pd.DataFrame(rows)

    logger.info("构造训练集 Group E1 特征...")
    train_feat = _build_group_e1_samples(old_df, "train", common_devices)
    logger.info("构造测试集 Group E1 特征...")
    test_feat = _build_group_e1_samples(new_df, None, common_devices)

    if train_feat.empty or test_feat.empty:
        logger.error("特征构造结果为空")
        return None

    meta_cols = {"device_id"}
    feature_cols = [c for c in train_feat.columns if c not in meta_cols]

    train_feat[feature_cols] = train_feat[feature_cols].replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0)
    test_feat[feature_cols] = test_feat[feature_cols].replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0)

    X_train = train_feat[feature_cols].values.astype(np.float32)
    y_train = le.transform(train_feat["device_id"])
    X_test = test_feat[feature_cols].values.astype(np.float32)
    y_test = le.transform(test_feat["device_id"])

    logger.info("训练: %d 样本, 测试: %d 样本", len(X_train), len(X_test))

    clf = _build_classifier(model_type)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)

    time_costs = {
        "train_time": clf.train_time,
        "inference_time_total": clf.inference_time_total,
        "inference_time_per_sample": clf.inference_time_per_sample,
    }

    inv_map = {v: k for k, v in dict(enumerate(le.classes_)).items()}
    save_dir = os.path.join(EXPR_DIR, tag)
    metrics = compute_metrics(y_test, y_pred, label_map=inv_map,
                              time_costs=time_costs, save_dir=save_dir, tag=tag)
    return metrics


def cross_dataset_group_d(rnn_type="lstm", seq_length=None, step=None):
    cfg = CROSS_GROUP_D_CONFIG[rnn_type]
    seq_length = seq_length or cfg["seq_length"]
    step = step or cfg["step"]
    tag = f"cross_dataset_group_d_{rnn_type}_H{seq_length}_L{step}"
    logger.info("=== %s ===", tag)

    checkpoint_path = os.path.join(
        RESULTS_DIR, "models", f"group_d_{rnn_type}_H{seq_length}_L{step}_best.pt"
    )
    if not os.path.exists(checkpoint_path):
        logger.error("未找到 Group D checkpoint: %s", checkpoint_path)
        return None

    old_df, le = _full_label_encoder()
    new_df = load_packet_csv(NEW_CSV_PATH)
    common_devices = sorted(
        set(old_df["device_id"].unique()) & set(new_df["device_id"].unique())
    )
    logger.info("共同设备 (%d): %s", len(common_devices), common_devices)
    if not common_devices:
        logger.error("无共同设备，跳过")
        return None

    # Refit the training-data scaler only; this does not retrain the RNN.
    loader = GroupDLoader()
    X_train, y_train, X_val, y_val, X_test_old, y_test_old, info = loader.load(
        seq_length=seq_length, step=step
    )

    new_subset = new_df[new_df["device_id"].isin(common_devices)]
    seqs, labels = _build_sequences(new_subset, seq_length, step, "test")
    arrays = []
    for seq_df in seqs:
        arrays.append(seq_df[GROUP_D_FEATURES].values.astype(np.float32))
    if arrays:
        X_new = np.stack(arrays, axis=0)
    else:
        logger.error("Group D 测试序列为空")
        return None

    n_feat = X_new.shape[-1]
    X_new = loader.scaler.transform(
        X_new.reshape(-1, n_feat)
    ).reshape(len(X_new), seq_length, n_feat).astype(np.float32)
    y_test = le.transform(labels)

    model = IoTRNNClassifier(
        input_size=len(GROUP_D_FEATURES),
        hidden_size=64,
        num_layers=2,
        num_classes=len(le.classes_),
        rnn_type=rnn_type,
        dropout=0.3,
    )
    trainer = RNNTrainer(model, device=DEVICE)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])

    _, _, test_loader = create_dataloaders(
        X_train[:1], y_train[:1], X_val[:1], y_val[:1], X_new, y_test, batch_size=32
    )
    t0 = time.perf_counter()
    _, _, y_true, y_pred = trainer.evaluate(test_loader, measure_latency=True)
    t1 = time.perf_counter()

    inference_total = t1 - t0
    inference_per_sample = inference_total / max(len(y_true), 1)
    time_costs = {
        "inference_time_total": inference_total,
        "inference_time_per_sample": inference_per_sample,
    }

    inv_map = {idx: name for idx, name in enumerate(le.classes_)}
    save_dir = os.path.join(EXPR_DIR, tag)
    metrics = compute_metrics(
        y_true, y_pred, label_map=inv_map, time_costs=time_costs,
        save_dir=save_dir, tag=tag,
    )
    return metrics


# ═══════════════════════════════════════════════════
# 全部跨数据集评估
# ═══════════════════════════════════════════════════

def run_cross_dataset_all():
    """运行全部跨数据集评估实验"""
    logger.info("===== 跨数据集评估: 解析新 PCAP =====")
    parse_new_pcap(force=True)

    results = []

    logger.info("===== 跨数据集评估: Group A =====")
    for mt in ("rf", "xgb"):
        m = cross_dataset_group_a(mt)
        if m:
            results.append({"group": "A", "model": mt, **{k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}})

    logger.info("===== 跨数据集评估: Group B =====")
    for mt in ("rf", "xgb"):
        m = cross_dataset_group_b(mt)
        if m:
            results.append({"group": "B", "model": mt, **{k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}})

    logger.info("===== 跨数据集评估: Group C =====")
    for mt in ("rf", "xgb"):
        m = cross_dataset_group_c(mt)
        if m:
            results.append({"group": "C", "model": mt, **{k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}})

    logger.info("===== 跨数据集评估: Group D =====")
    for rt in ("lstm", "gru"):
        m = cross_dataset_group_d(rt)
        if m:
            results.append({"group": "D", "model": rt, **{k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}})

    logger.info("===== 跨数据集评估: Group E0 =====")
    for mt in ("rf", "xgb"):
        m = cross_dataset_group_e(mt)
        if m:
            results.append({"group": "E0", "model": mt, **{k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}})

    logger.info("===== 跨数据集评估: Group E1 =====")
    for mt in ("rf", "xgb"):
        m = cross_dataset_group_e1(mt)
        if m:
            results.append({"group": "E1", "model": mt, **{k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")}})

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(TABLE_DIR, "cross_dataset_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info("跨数据集结果汇总:\n%s", df.to_string(index=False))
        logger.info("已保存: %s", csv_path)

    logger.info("===== 跨数据集评估完毕 =====")
