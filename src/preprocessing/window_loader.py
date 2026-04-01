"""
Group B: 窗口级数据加载器
一个样本 = 一个时间窗口，按 device_id 分组、按 ts 排序后切分
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
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STEP, MIN_PACKETS_PER_WINDOW,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


class GroupBLoader:

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}

    def load(self, csv_path=None, window_size=DEFAULT_WINDOW_SIZE,
             step=DEFAULT_WINDOW_STEP):
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
                if len(ts_arr) == 0:
                    continue

                t_start = ts_arr[0]
                t_end = ts_arr[-1]
                win_start = t_start

                while win_start < t_end:
                    win_end = win_start + window_size
                    mask = (ts_arr >= win_start) & (ts_arr < win_end)
                    win_df = dev_df.loc[mask]

                    if len(win_df) >= MIN_PACKETS_PER_WINDOW:
                        feat = compute_window_features(win_df)
                        feat["device_id"] = device_id
                        feat["split"] = split_name
                        all_rows.append(feat)

                    win_start += step

        feat_df = pd.DataFrame(all_rows)
        feat_time = time.time() - t0

        if feat_df.empty:
            raise RuntimeError("窗口构造结果为空")

        # 保存窗口级样本表
        save_path = os.path.join(
            PROCESSED_DIR, f"window_samples_W{window_size}_S{step}.csv")
        feat_df.to_csv(save_path, index=False)
        logger.info("窗口级样本表已保存: %s (%d 行)", save_path, len(feat_df))

        # 特征列名（排除标识列）
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
            "window_size": window_size,
            "window_step": step,
            "feature_construction_time": feat_time,
        }
        logger.info("Group B 加载完毕 (W=%d, S=%d): train=%d, val=%d, test=%d",
                     window_size, step,
                     info["train_samples"], info["val_samples"], info["test_samples"])
        return X_train, y_train, X_val, y_val, X_test, y_test, info
