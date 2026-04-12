"""
Group E1: 关系增强混合粒度数据加载器
每条样本 = 当前包即时状态子集 + 历史行为基线子集 + 关系特征 + 阶段化冷启动特征

与 E0（GroupELoader）的区别：
1. 当前包选择：向锚点回溯 D_max 秒，优先选有载荷的包
2. 特征精选：不全量拼接 A+B，而是保留互补子集
3. 关系特征：当前包相对历史基线的偏离（z-score / diff / match）
4. 阶段化冷启动：cold / partial / full 三阶段
"""
import logging
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.pcap_loader import load_packet_csv
from src.features.window_features import compute_window_features
from src.utils.config import (
    DEFAULT_GROUP_E_SAMPLE_STEP, DEFAULT_GROUP_E_HIST_WINDOW,
    DEFAULT_GROUP_E1_D_MAX, PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

# ── 当前包即时状态特征（Group A 子集） ──────────
CUR_PACKET_FIELDS = [
    "packet_len", "payload_len", "payload_entropy", "delta_t",
    "ip_proto", "tcp_syn", "tcp_ack", "tcp_fin", "tcp_psh", "direction",
]

# ── 历史行为基线（Group B 子集，从 compute_window_features 结果中选取） ──
HIST_BASELINE_KEYS = [
    "iat_mean", "iat_std", "iat_max",
    "packet_len_mean", "packet_len_min", "packet_len_max", "packet_len_std",
    "payload_len_mean", "payload_len_std",
    "payload_entropy_mean", "payload_entropy_std",
    "unique_peer_ip_count",
    "https_port_ratio", "well_known_dst_port_ratio",
    "tcp_ratio", "udp_ratio",
    "pkt_count", "total_bytes",
]


def _safe_div(a, b, default=0.0):
    return a / b if abs(b) > 1e-9 else default


def _compute_relation_features(cur_pkt, win_stats, hist_df):
    """计算当前包相对历史基线的关系特征"""
    rel = {}

    # 包长偏离
    h_mean = win_stats.get("packet_len_mean", 0.0)
    h_std = win_stats.get("packet_len_std", 0.0)
    c_pktlen = float(cur_pkt.get("packet_len", 0.0))
    rel["rel_pktlen_diff"] = c_pktlen - h_mean
    rel["rel_pktlen_zscore"] = _safe_div(c_pktlen - h_mean, max(h_std, 1e-6))
    rel["rel_pktlen_ratio"] = _safe_div(c_pktlen, max(h_mean, 1.0))

    # 载荷长偏离
    hp_mean = win_stats.get("payload_len_mean", 0.0)
    hp_std = win_stats.get("payload_len_std", 0.0)
    c_paylen = float(cur_pkt.get("payload_len", 0.0))
    rel["rel_payloadlen_diff"] = c_paylen - hp_mean
    rel["rel_payloadlen_zscore"] = _safe_div(c_paylen - hp_mean, max(hp_std, 1e-6))

    # IAT 偏离
    hi_mean = win_stats.get("iat_mean", 0.0)
    hi_std = win_stats.get("iat_std", 0.0)
    c_iat = float(cur_pkt.get("delta_t", 0.0))
    rel["rel_iat_diff"] = c_iat - hi_mean
    rel["rel_iat_zscore"] = _safe_div(c_iat - hi_mean, max(hi_std, 1e-6))
    rel["rel_iat_ratio"] = _safe_div(c_iat, max(hi_mean, 1e-6))

    # 熵偏离
    he_mean = win_stats.get("payload_entropy_mean", 0.0)
    c_ent = float(cur_pkt.get("payload_entropy", 0.0))
    rel["rel_entropy_diff"] = c_ent - he_mean

    # 协议匹配
    c_proto = int(cur_pkt.get("ip_proto", 0))
    if len(hist_df) > 0:
        major_proto = int(hist_df["ip_proto"].mode().iloc[0])
        rel["rel_protocol_match"] = 1.0 if c_proto == major_proto else 0.0
    else:
        rel["rel_protocol_match"] = 0.0

    # 方向匹配
    c_dir = int(cur_pkt.get("direction", 0))
    if len(hist_df) > 0:
        major_dir = int(hist_df["direction"].mode().iloc[0])
        rel["rel_direction_match"] = 1.0 if c_dir == major_dir else 0.0
    else:
        rel["rel_direction_match"] = 0.0

    # 目标端口是否在历史 top-3
    c_dport = int(cur_pkt.get("dst_port", 0))
    if len(hist_df) > 0:
        top_ports = set(hist_df["dst_port"].value_counts().head(3).index)
        rel["rel_dst_port_in_topk"] = 1.0 if c_dport in top_ports else 0.0
    else:
        rel["rel_dst_port_in_topk"] = 0.0

    return rel


class GroupE1Loader:

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}

    def load(self, csv_path=None,
             sample_step=DEFAULT_GROUP_E_SAMPLE_STEP,
             hist_window=DEFAULT_GROUP_E_HIST_WINDOW,
             d_max=DEFAULT_GROUP_E1_D_MAX):
        """
        返回 (X_train, y_train, X_val, y_val, X_test, y_test, info)
        """
        t0 = time.time()
        df = load_packet_csv(csv_path) if csv_path else load_packet_csv()

        self.label_encoder.fit(df["device_id"].unique())
        self.label_map = {
            name: idx for idx, name in enumerate(self.label_encoder.classes_)
        }

        all_rows = []

        for split_name in ["train", "val", "test"]:
            split_df = df[df["split"] == split_name]

            for device_id, dev_df in split_df.groupby("device_id"):
                dev_df = dev_df.sort_values("ts").reset_index(drop=True)
                ts_arr = dev_df["ts"].values
                payload_arr = dev_df["payload_len"].values

                if len(ts_arr) < 2:
                    continue

                t_start = ts_arr[0]
                t_end = ts_arr[-1]

                anchor = t_start + sample_step
                while anchor <= t_end:
                    # ── 方案 C: 向锚点回溯选包 ──────────
                    cur_idx = self._select_packet(
                        ts_arr, payload_arr, anchor, d_max)

                    if cur_idx is None:
                        anchor += sample_step
                        continue

                    t_cur = ts_arr[cur_idx]
                    cur_pkt = dev_df.iloc[cur_idx]
                    anchor_gap = anchor - t_cur

                    # ── (a) 当前包即时状态 ───────────────
                    cur_feat = {}
                    for f in CUR_PACKET_FIELDS:
                        cur_feat[f"cur_{f}"] = float(cur_pkt.get(f, 0.0))

                    # ── 历史窗口 (t_cur - W, t_cur) ─────
                    hist_start = t_cur - hist_window
                    hist_mask = (ts_arr >= hist_start) & (ts_arr < t_cur)
                    hist_df = dev_df.loc[hist_mask]
                    n_hist = len(hist_df)

                    # 计算窗口统计
                    if n_hist >= 1:
                        win_stats = compute_window_features(hist_df)
                    else:
                        win_stats = {k: 0.0 for k in HIST_BASELINE_KEYS}

                    # ── (b) 历史行为基线子集 ─────────────
                    hist_feat = {}
                    for k in HIST_BASELINE_KEYS:
                        hist_feat[f"hist_{k}"] = float(win_stats.get(k, 0.0))

                    # 计算 packet_rate / byte_rate
                    if n_hist >= 2:
                        actual_dur = float(
                            ts_arr[hist_mask][-1] - ts_arr[hist_mask][0])
                    else:
                        actual_dur = 0.0
                    safe_dur = max(actual_dur, 1e-6)
                    hist_feat["hist_packet_rate"] = n_hist / safe_dur
                    hist_feat["hist_byte_rate"] = float(
                        win_stats.get("total_bytes", 0.0)) / safe_dur

                    # ── (c) 关系特征 ─────────────────────
                    rel_feat = _compute_relation_features(
                        cur_pkt, win_stats, hist_df)

                    # ── (d) 历史可靠性与阶段 ─────────────
                    ratio = actual_dur / hist_window if hist_window > 0 else 0.0
                    if ratio < 0.3:
                        stage = 0  # cold
                    elif ratio < 0.8:
                        stage = 1  # partial
                    else:
                        stage = 2  # full

                    reliability_feat = {
                        "hist_duration_actual": actual_dur,
                        "hist_window_ratio": ratio,
                        "hist_packet_count_actual": float(n_hist),
                        "hist_stage": float(stage),
                        "anchor_gap": anchor_gap,
                    }

                    # ── 合并 ─────────────────────────────
                    row = {}
                    row.update(cur_feat)
                    row.update(hist_feat)
                    row.update(rel_feat)
                    row.update(reliability_feat)
                    row["device_id"] = device_id
                    row["split"] = split_name
                    all_rows.append(row)

                    anchor += sample_step

        feat_df = pd.DataFrame(all_rows)
        feat_time = time.time() - t0

        if feat_df.empty:
            raise RuntimeError("Group E1 样本构造结果为空")

        # 保存中间样本表
        save_path = os.path.join(
            PROCESSED_DIR,
            f"group_e1_samples_S{sample_step}_W{hist_window}.csv")
        feat_df.to_csv(save_path, index=False)
        logger.info("Group E1 样本表已保存: %s (%d 行)", save_path, len(feat_df))

        # 特征列名
        meta_cols = {"device_id", "split"}
        feature_cols = [c for c in feat_df.columns if c not in meta_cols]

        feat_df["label"] = self.label_encoder.transform(feat_df["device_id"])
        feat_df[feature_cols] = feat_df[feature_cols].fillna(0)

        # 替换 inf
        feat_df[feature_cols] = feat_df[feature_cols].replace(
            [np.inf, -np.inf], 0.0)

        train = feat_df[feat_df["split"] == "train"]
        val = feat_df[feat_df["split"] == "val"]
        test = feat_df[feat_df["split"] == "test"]

        X_train = train[feature_cols].values.astype(np.float32)
        y_train = train["label"].values
        X_val = val[feature_cols].values.astype(np.float32)
        y_val = val["label"].values
        X_test = test[feature_cols].values.astype(np.float32)
        y_test = test["label"].values

        info = {
            "feature_names": feature_cols,
            "label_map": self.label_map,
            "n_classes": len(self.label_map),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "sample_step": sample_step,
            "hist_window": hist_window,
            "feature_construction_time": feat_time,
        }
        logger.info(
            "Group E1 加载完毕 (S=%d, W=%d): train=%d, val=%d, test=%d, "
            "特征维度=%d",
            sample_step, hist_window,
            info["train_samples"], info["val_samples"], info["test_samples"],
            len(feature_cols))
        return X_train, y_train, X_val, y_val, X_test, y_test, info

    @staticmethod
    def _select_packet(ts_arr, payload_arr, t_ref, d_max):
        """方案 C: 在 [t_ref - d_max, t_ref] 内选包，优先有载荷"""
        # 找 ts <= t_ref 的最后一个包
        right = np.searchsorted(ts_arr, t_ref, side="right") - 1
        if right < 0:
            return None

        # 在 [t_ref - d_max, t_ref] 范围内搜索
        t_low = t_ref - d_max
        best_payload_idx = None
        best_any_idx = None

        for i in range(right, -1, -1):
            if ts_arr[i] < t_low:
                break
            if best_any_idx is None:
                best_any_idx = i
            if payload_arr[i] > 0 and best_payload_idx is None:
                best_payload_idx = i
                break  # 找到最近的有载荷包即停

        if best_payload_idx is not None:
            return best_payload_idx
        return best_any_idx  # 可能是 None（区间内无包）
