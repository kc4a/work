"""
Group E: 混合粒度数据加载器
每条样本 = 一个"当前包"的单包特征 + 历史窗口行为统计 + 冷启动补偿特征
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
    GROUP_A_FEATURES, GROUP_E_COLD_START_FEATURES,
    DEFAULT_GROUP_E_SAMPLE_STEP, DEFAULT_GROUP_E_HIST_WINDOW,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

# 空历史时窗口特征的默认 key 列表（由第一次成功计算确定）
_WINDOW_FEATURE_KEYS: list = []


def _get_empty_window_features() -> dict:
    """返回全 0 的窗口特征字典（key 列表在首次计算时缓存）"""
    return {k: 0.0 for k in _WINDOW_FEATURE_KEYS}


def _init_window_feature_keys(sample_df: pd.DataFrame):
    """用一个小样本调用 compute_window_features 获取完整 key 列表"""
    global _WINDOW_FEATURE_KEYS
    if _WINDOW_FEATURE_KEYS:
        return
    feat = compute_window_features(sample_df)
    _WINDOW_FEATURE_KEYS = list(feat.keys())


class GroupELoader:

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}

    def load(self, csv_path=None,
             sample_step=DEFAULT_GROUP_E_SAMPLE_STEP,
             hist_window=DEFAULT_GROUP_E_HIST_WINDOW):
        """
        返回 (X_train, y_train, X_val, y_val, X_test, y_test, info)
        """
        t0 = time.time()
        df = load_packet_csv(csv_path) if csv_path else load_packet_csv()

        self.label_encoder.fit(df["device_id"].unique())
        self.label_map = {
            name: idx for idx, name in enumerate(self.label_encoder.classes_)
        }

        # 初始化窗口特征 key 列表（用前 10 个包）
        _init_window_feature_keys(df.head(10))

        all_rows = []

        for split_name in ["train", "val", "test"]:
            split_df = df[df["split"] == split_name]

            for device_id, dev_df in split_df.groupby("device_id"):
                dev_df = dev_df.sort_values("ts").reset_index(drop=True)
                ts_arr = dev_df["ts"].values

                if len(ts_arr) < 2:
                    continue

                t_start = ts_arr[0]
                t_end = ts_arr[-1]

                # 生成锚点: t_start + S, t_start + 2S, ...
                anchor = t_start + sample_step
                while anchor <= t_end:
                    # 选择 ts >= anchor 的第一个包作为当前包
                    idx = np.searchsorted(ts_arr, anchor, side="left")
                    if idx >= len(ts_arr):
                        break

                    t_cur = ts_arr[idx]
                    cur_pkt = dev_df.iloc[idx]

                    # --- 当前包特征 (Group A) ---
                    pkt_feat = {}
                    for f in GROUP_A_FEATURES:
                        pkt_feat[f"pkt_{f}"] = cur_pkt.get(f, 0.0)

                    # --- 历史窗口 (t_cur - W, t_cur) ---
                    hist_start = t_cur - hist_window
                    hist_mask = (ts_arr >= hist_start) & (ts_arr < t_cur)
                    hist_df = dev_df.loc[hist_mask]
                    n_hist = len(hist_df)

                    if n_hist >= 1:
                        win_feat = compute_window_features(hist_df)
                    else:
                        win_feat = _get_empty_window_features()

                    # 给窗口特征加前缀避免与包特征冲突
                    hist_feat = {f"hist_{k}": v for k, v in win_feat.items()}

                    # --- 冷启动补偿特征 ---
                    if n_hist >= 2:
                        actual_dur = float(ts_arr[hist_mask][-1] - ts_arr[hist_mask][0])
                    elif n_hist == 1:
                        actual_dur = 0.0
                    else:
                        actual_dur = 0.0

                    total_hist_bytes = float(hist_df["packet_len"].sum()) if n_hist > 0 else 0.0
                    safe_dur = max(actual_dur, 1e-6)

                    cold_feat = {
                        "hist_duration_actual": actual_dur,
                        "hist_window_ratio": actual_dur / hist_window,
                        "hist_complete_flag": 1.0 if actual_dur >= hist_window * 0.9 else 0.0,
                        "hist_packet_count_actual": float(n_hist),
                        "packet_rate": n_hist / safe_dur,
                        "byte_rate": total_hist_bytes / safe_dur,
                    }

                    # --- 合并 ---
                    row = {}
                    row.update(pkt_feat)
                    row.update(hist_feat)
                    row.update(cold_feat)
                    row["device_id"] = device_id
                    row["split"] = split_name
                    all_rows.append(row)

                    anchor += sample_step

        feat_df = pd.DataFrame(all_rows)
        feat_time = time.time() - t0

        if feat_df.empty:
            raise RuntimeError("Group E 样本构造结果为空")

        # 保存中间样本表
        save_path = os.path.join(
            PROCESSED_DIR, f"group_e_samples_S{sample_step}_W{hist_window}.csv")
        feat_df.to_csv(save_path, index=False)
        logger.info("Group E 样本表已保存: %s (%d 行)", save_path, len(feat_df))

        # 特征列名
        meta_cols = {"device_id", "split"}
        feature_cols = [c for c in feat_df.columns if c not in meta_cols]

        feat_df["label"] = self.label_encoder.transform(feat_df["device_id"])
        feat_df[feature_cols] = feat_df[feature_cols].fillna(0)

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
            "Group E 加载完毕 (S=%d, W=%d): train=%d, val=%d, test=%d, 特征维度=%d",
            sample_step, hist_window,
            info["train_samples"], info["val_samples"], info["test_samples"],
            len(feature_cols))
        return X_train, y_train, X_val, y_val, X_test, y_test, info
