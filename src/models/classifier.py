"""
分类器封装
"""
import time
import pickle
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

from ..utils.config import RF_PARAMS, XGB_PARAMS, RANDOM_STATE, INFERENCE_REPEAT, RESULTS_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)

# 尝试导入XGBoost，如果没有则使用RF
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available, will use RandomForest only")


class IoTClassifier:
    """IoT设备分类器"""
    
    def __init__(self, model_type: str = 'rf', **kwargs):
        """
        初始化分类器
        
        Args:
            model_type: 'rf' 或 'xgb'
            **kwargs: 额外的模型参数
        """
        self.model_type = model_type.lower()
        self.model = None
        self.train_time = 0.0
        self.inference_times = []
        
        if self.model_type == 'rf':
            params = {**RF_PARAMS, **kwargs}
            self.model = RandomForestClassifier(**params)
        elif self.model_type == 'xgb':
            if not XGB_AVAILABLE:
                logger.warning("XGBoost not available, falling back to RF")
                self.model_type = 'rf'
                params = {**RF_PARAMS, **kwargs}
                self.model = RandomForestClassifier(**params)
            else:
                params = {**XGB_PARAMS, **kwargs}
                self.model = xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Initialized {self.model_type} classifier")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'IoTClassifier':
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            
        Returns:
            self
        """
        logger.info(f"Training {self.model_type} model...")
        
        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        t1 = time.perf_counter()
        
        self.train_time = t1 - t0
        logger.info(f"Training completed in {self.train_time:.2f}s")
        
        return self
    
    def predict(self, X_test: np.ndarray, warmup: bool = True) -> np.ndarray:
        """预测并测量推理延迟"""
        if warmup and len(X_test) > 0:
            _ = self.model.predict(X_test[:1])

        t0 = time.perf_counter()
        y_pred = self.model.predict(X_test)
        t1 = time.perf_counter()
        self.inference_time_total = t1 - t0
        self.inference_time_per_sample = (
            self.inference_time_total / len(X_test) if len(X_test) > 0 else 0.0
        )

        # 单样本延迟统计
        n_repeat = min(INFERENCE_REPEAT, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_repeat, replace=False)
        self.inference_times = []
        for idx in sample_indices:
            x_sample = X_test[idx:idx + 1]
            ts = time.perf_counter()
            _ = self.model.predict(x_sample)
            te = time.perf_counter()
            self.inference_times.append((te - ts) * 1000)

        avg_latency = np.mean(self.inference_times)
        logger.info(f"Inference: total={self.inference_time_total:.3f}s, "
                     f"per_sample={avg_latency:.4f}ms")
        return y_pred
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'IoTClassifier':
        """加载模型"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return self
    
    def get_model_size_kb(self) -> float:
        """获取模型大小（KB）"""
        import tempfile
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                tmp_file = tmp.name
            self.save(tmp_file)
            size_kb = os.path.getsize(tmp_file) / 1024
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except PermissionError:
                    pass  # Windows may have file lock issues
        return size_kb


def get_inference_latency(model, X_sample: np.ndarray, n_repeat: int = 100) -> float:
    """
    测量模型推理延迟
    
    Args:
        model: 训练好的模型
        X_sample: 样本数据
        n_repeat: 重复次数
        
    Returns:
        平均延迟（毫秒）
    """
    times = []
    for _ in range(n_repeat):
        idx = np.random.randint(0, len(X_sample))
        x = X_sample[idx:idx+1]
        
        t0 = time.perf_counter()
        _ = model.predict(x)
        t1 = time.perf_counter()
        
        times.append((t1 - t0) * 1000)
    
    return float(np.mean(times))


if __name__ == "__main__":
    # 测试
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                               n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 测试RF
    clf = IoTClassifier('rf')
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\nRF Accuracy: {(y_pred == y_test).mean():.4f}")
    
    print(f"Train time: {clf.train_time:.2f}s")
    print(f"Inference total: {clf.inference_time_total:.4f}s")
