"""
RNN分类器模块 - LSTM / GRU

用于序列特征分类
"""
import time
import numpy as np
from typing import Tuple, Dict, List, Optional
import os

# 尝试导入torch，如果失败则设置标志
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..utils.config import (
    RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, RNN_DROPOUT, 
    RNN_BATCH_SIZE, RNN_LEARNING_RATE, RNN_EPOCHS, RNN_PATIENCE,
    RESULTS_DIR
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

if not TORCH_AVAILABLE:
    logger.warning("PyTorch not available. RNN models will not work.")


class IoTSequenceDataset(torch.utils.data.Dataset if TORCH_AVAILABLE else object):
    """IoT序列数据集（PyTorch Dataset包装）"""

    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        """
        Args:
            X_seq: shape (N, seq_len, feat_dim)
            y_seq: shape (N,)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RNN models")

        super().__init__()
        self.X = torch.FloatTensor(X_seq)
        self.y = torch.LongTensor(y_seq)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IoTRNNClassifier:
    """IoT RNN分类器 (LSTM/GRU) - 包装类"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = RNN_HIDDEN_SIZE,
                 num_layers: int = RNN_NUM_LAYERS,
                 num_classes: int = 10,
                 rnn_type: str = 'lstm',
                 dropout: float = RNN_DROPOUT):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: RNN隐藏层大小
            num_layers: RNN层数
            num_classes: 类别数
            rnn_type: 'lstm' 或 'gru'
            dropout: dropout率
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RNN models. "
                            "Please install: pip install torch")
        
        self.model = _IoTRNNModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            rnn_type=rnn_type,
            dropout=dropout
        )
        
        logger.info(f"Initialized {rnn_type.upper()} model: "
                   f"input={input_size}, hidden={hidden_size}, "
                   f"layers={num_layers}, classes={num_classes}")
    
    def count_parameters(self):
        """统计可训练参数数量"""
        return self.model.count_parameters()


class _IoTRNNModel(torch.nn.Module if TORCH_AVAILABLE else object):
    """实际的RNN模型实现"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = RNN_HIDDEN_SIZE,
                 num_layers: int = RNN_NUM_LAYERS,
                 num_classes: int = 10,
                 rnn_type: str = 'lstm',
                 dropout: float = RNN_DROPOUT):
        if not TORCH_AVAILABLE:
            return
        
        super(_IoTRNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # RNN层
        if self.rnn_type == 'lstm':
            self.rnn = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        elif self.rnn_type == 'gru':
            self.rnn = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        
        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        # RNN输出
        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(x)
        else:  # GRU
            output, hidden = self.rnn(x)
        
        # 取最后一个时间步的隐藏状态
        last_hidden = output[:, -1, :]
        
        # 分类
        logits = self.classifier(last_hidden)
        
        return logits
    
    def count_parameters(self):
        """统计可训练参数数量"""
        if not TORCH_AVAILABLE:
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RNNTrainer:
    """RNN训练器"""
    
    def __init__(self, 
                 model: IoTRNNClassifier,
                 device = None,
                 lr: float = RNN_LEARNING_RATE):
        """
        Args:
            model: RNN模型
            device: 计算设备
            lr: 学习率
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RNN training")
        
        self.model = model.model
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        self.inference_times = []
        
        logger.info(f"Trainer initialized on {self.device}")
    
    def train_epoch(self, dataloader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, measure_latency: bool = False):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        self.inference_times = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 测量推理时间
                if measure_latency:
                    # 预热
                    _ = self.model(batch_X[:1])
                    
                    # 计时
                    start = time.perf_counter()
                    outputs = self.model(batch_X)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    
                    batch_time = (end - start) * 1000  # ms
                    self.inference_times.extend([batch_time / len(batch_X)] * len(batch_X))
                else:
                    outputs = self.model(batch_X)
                
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)
    
    def train(self, 
              train_loader,
              val_loader,
              epochs: int = RNN_EPOCHS,
              patience: int = RNN_PATIENCE,
              save_path: str = None):
        """完整训练流程"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, "models", "best_rnn_model.pt")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_start_time = time.time()
        
        logger.info(f"Starting training: max_epochs={epochs}, patience={patience}")
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            # 记录
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                           f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            
            # Early Stopping检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最优模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        train_time = time.time() - train_start_time
        
        # 加载最优模型
        checkpoint = torch.load(save_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Training completed in {train_time:.2f}s")
        logger.info(f"Best val_acc: {checkpoint['val_acc']:.4f} at epoch {checkpoint['epoch']}")
        
        return {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_epoch': checkpoint['epoch'],
            'best_val_acc': checkpoint['val_acc'],
            'train_time': train_time
        }
    
    def predict_single(self, x: np.ndarray):
        """单样本预测（用于精确计时）"""
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # 预热
            _ = self.model(x_tensor)
            
            # 精确计时
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            output = self.model(x_tensor)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            _, predicted = torch.max(output, 1)
        
        inference_time = (end - start) * 1000  # ms
        
        return predicted.item(), inference_time
    
    def get_model_size_kb(self) -> float:
        """获取模型大小（KB）"""
        save_path = os.path.join(RESULTS_DIR, "models", "temp_model_size.pt")
        torch.save(self.model.state_dict(), save_path)
        size_kb = os.path.getsize(save_path) / 1024
        os.remove(save_path)
        return size_kb


def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None,
                      X_test: np.ndarray = None, y_test: np.ndarray = None,
                      batch_size: int = RNN_BATCH_SIZE,
                      num_workers: int = 0):
    """
    创建数据加载器（支持 train / val / test 三个划分）

    Returns:
        (train_loader, val_loader, test_loader)
        如果 val 或 test 为 None 则对应 loader 为 None
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    pin = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        IoTSequenceDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )

    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_loader = torch.utils.data.DataLoader(
            IoTSequenceDataset(X_val, y_val),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
        )

    test_loader = None
    if X_test is not None and y_test is not None and len(X_test) > 0:
        test_loader = torch.utils.data.DataLoader(
            IoTSequenceDataset(X_test, y_test),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        # 测试模型
        model = IoTRNNClassifier(
            input_size=19,
            hidden_size=64,
            num_layers=2,
            num_classes=10,
            rnn_type='lstm'
        )
        
        print(f"Model parameters: {model.count_parameters():,}")
        
        # 测试前向传播
        x = torch.randn(4, 10, 19)  # (batch, seq_len, feat_dim)
        output = model.model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    else:
        print("PyTorch not available. Cannot run tests.")
