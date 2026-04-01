"""
Group C: 序列压缩特征 + ML
Group D: 原始序列级 + RNN
共用相同的序列构造逻辑，按 device_id 分组、按 ts 排序、按固定包数切分
"""
import logging
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.preprocessing.pcap_loader import load_packet_csv
from src.features.sequence_compressed_features import compute_sequence_compressed_features
from src.utils.config import (
    DEFAULT_SEQ_LENGTH, DEFAULT_SEQ_STEP, GROUP_D_FEATURES, PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def _build_sequences(df, seq_length, step, split_name):
    """在指定 split 内按 device 构建定长序列，返回 (list_of_seq_df, labels)"""
    split_df = df[df["split"] == split_name]
    seqs = []
    labels = []

    for device_id, dev_df in split_df.groupby("device_id"):
        dev_df = dev_df.sort_values("ts").reset_index(drop=True)
        n = len(dev_df)
        start = 0
        while start + seq_length <= n:
            seq_df = dev_df.iloc[start:start + seq_length]
            seqs.append(seq_df)
            labels.append(device_id)
            start += step

    return seqs, labels


class GroupCLoader:
    """序列压缩特征 + ML"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}

    def load(self, csv_path=None, seq_length=DEFAULT_SEQ_LENGTH,
             step=DEFAULT_SEQ_STEP):
        t0 = time.time()
        df = load_packet_csv(csv_path) if csv_path else load_packet_csv()

        self.label_encoder.fit(df["device_id"].unique())
        self.label_map = {
            name: idx for idx, name in enumerate(self.label_encoder.classes_)
        }

        results = {}
        all_save_rows = []
        for split_name in ["train", "val", "test"]:
            seqs, labels = _build_sequences(df, seq_length, step, split_name)

            rows = []
            for i, seq_df in enumerate(seqs):
                feat = compute_sequence_compressed_features(seq_df)
                rows.append(feat)
                all_save_rows.append({**feat, "device_id": labels[i], "split": split_name})

            feat_df = pd.DataFrame(rows).fillna(0)
            feature_cols = list(feat_df.columns)
            X = feat_df.values.astype(np.float32)
            y = self.label_encoder.transform(labels)
            results[split_name] = (X, y)

        feat_time = time.time() - t0

        # 保存序列压缩样本表
        save_path = os.path.join(
            PROCESSED_DIR, f"seq_compressed_samples_H{seq_length}_L{step}.csv")
        pd.DataFrame(all_save_rows).to_csv(save_path, index=False)
        logger.info("序列压缩样本表已保存: %s", save_path)

        X_train, y_train = results["train"]
        X_val, y_val = results["val"]
        X_test, y_test = results["test"]

        info = {
            "feature_names": feature_cols,
            "label_map": self.label_map,
            "n_classes": len(self.label_map),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "seq_length": seq_length,
            "seq_step": step,
            "feature_construction_time": feat_time,
        }
        logger.info("Group C 加载完毕 (H=%d, L=%d): train=%d, val=%d, test=%d",
                     seq_length, step,
                     info["train_samples"], info["val_samples"], info["test_samples"])
        return X_train, y_train, X_val, y_val, X_test, y_test, info


class GroupDLoader:
    """原始序列级 + RNN"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}
        self.scaler = StandardScaler()

    def load(self, csv_path=None, seq_length=DEFAULT_SEQ_LENGTH,
             step=DEFAULT_SEQ_STEP):
        t0 = time.time()
        df = load_packet_csv(csv_path) if csv_path else load_packet_csv()

        self.label_encoder.fit(df["device_id"].unique())
        self.label_map = {
            name: idx for idx, name in enumerate(self.label_encoder.classes_)
        }

        results = {}
        all_labels_for_save = {}
        for split_name in ["train", "val", "test"]:
            seqs, labels = _build_sequences(df, seq_length, step, split_name)

            arrays = []
            for seq_df in seqs:
                arr = seq_df[GROUP_D_FEATURES].values.astype(np.float32)
                arrays.append(arr)

            if arrays:
                X = np.stack(arrays, axis=0)  # (N, seq_length, n_features)
            else:
                X = np.empty((0, seq_length, len(GROUP_D_FEATURES)), dtype=np.float32)

            y = self.label_encoder.transform(labels)
            results[split_name] = (X, y)
            all_labels_for_save[split_name] = labels

        feat_time = time.time() - t0

        X_train, y_train = results["train"]
        X_val, y_val = results["val"]
        X_test, y_test = results["test"]

        # 保存序列级张量与索引
        save_dir = os.path.join(PROCESSED_DIR, f"seq_raw_H{seq_length}_L{step}")
        os.makedirs(save_dir, exist_ok=True)
        for split_name in ["train", "val", "test"]:
            X_s, y_s = results[split_name]
            np.save(os.path.join(save_dir, f"X_{split_name}.npy"), X_s)
            np.save(os.path.join(save_dir, f"y_{split_name}.npy"), y_s)
            pd.DataFrame({
                "seq_id": range(len(y_s)),
                "device_id": all_labels_for_save[split_name],
                "label": y_s,
            }).to_csv(os.path.join(save_dir, f"index_{split_name}.csv"), index=False)
        logger.info("序列级张量已保存: %s", save_dir)

        # 标准化：fit on train，transform all
        X_train, X_val, X_test = self.normalize(X_train, X_val, X_test)

        info = {
            "feature_names": GROUP_D_FEATURES,
            "label_map": self.label_map,
            "n_classes": len(self.label_map),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "seq_length": seq_length,
            "seq_step": step,
            "n_features_per_step": len(GROUP_D_FEATURES),
            "feature_construction_time": feat_time,
        }
        logger.info("Group D 加载完毕 (H=%d, L=%d): train=%d, val=%d, test=%d, shape=%s",
                     seq_length, step,
                     info["train_samples"], info["val_samples"], info["test_samples"],
                     X_train.shape)
        return X_train, y_train, X_val, y_val, X_test, y_test, info

    def normalize(self, X_train, X_val, X_test):
        """对 3D 数据做 StandardScaler（fit on train）"""
        n_train, seq_len, n_feat = X_train.shape

        flat_train = X_train.reshape(-1, n_feat)
        self.scaler.fit(flat_train)

        X_train = self.scaler.transform(flat_train).reshape(n_train, seq_len, n_feat).astype(np.float32)

        if len(X_val) > 0:
            X_val = self.scaler.transform(
                X_val.reshape(-1, n_feat)
            ).reshape(len(X_val), seq_len, n_feat).astype(np.float32)

        if len(X_test) > 0:
            X_test = self.scaler.transform(
                X_test.reshape(-1, n_feat)
            ).reshape(len(X_test), seq_len, n_feat).astype(np.float32)

        return X_train, X_val, X_test
