"""
Group A: 单包级数据加载器
一个样本 = 一个包, 28 个特征
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.pcap_loader import load_packet_csv
from src.utils.config import GROUP_A_FEATURES

logger = logging.getLogger(__name__)


class GroupALoader:

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_map = {}

    def load(self, csv_path=None):
        """
        返回 (X_train, y_train, X_val, y_val, X_test, y_test, info)
        """
        df = load_packet_csv(csv_path) if csv_path else load_packet_csv()

        # 编码标签
        self.label_encoder.fit(df["device_id"])
        self.label_map = {
            name: idx for idx, name in enumerate(self.label_encoder.classes_)
        }
        df["label"] = self.label_encoder.transform(df["device_id"])

        # 填充 NaN
        for col in GROUP_A_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 按 split 划分
        train = df[df["split"] == "train"]
        val = df[df["split"] == "val"]
        test = df[df["split"] == "test"]

        X_train = train[GROUP_A_FEATURES].values.astype(np.float32)
        y_train = train["label"].values
        X_val = val[GROUP_A_FEATURES].values.astype(np.float32)
        y_val = val["label"].values
        X_test = test[GROUP_A_FEATURES].values.astype(np.float32)
        y_test = test["label"].values

        info = {
            "feature_names": GROUP_A_FEATURES,
            "label_map": self.label_map,
            "n_classes": len(self.label_map),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }
        logger.info("Group A 加载完毕: train=%d, val=%d, test=%d, classes=%d",
                     info["train_samples"], info["val_samples"],
                     info["test_samples"], info["n_classes"])
        return X_train, y_train, X_val, y_val, X_test, y_test, info
