# 基于多粒度网络行为表征的 IoT 设备识别

本项目为毕业论文实验框架，研究不同粒度的网络流量表征对 IoT 设备识别效果的影响。通过将原始 PCAP 流量解析为统一的包级数据，再分别构建四种粒度的特征表征，配合传统机器学习与深度学习模型进行对比实验。

## 实验设计

| 实验组 | 粒度 | 特征维度 | 分类器 |
|--------|------|----------|--------|
| **A** | 单包级 | 28 维包头特征 | Random Forest / XGBoost |
| **B** | 时间窗口级 | ~55 维统计特征 | Random Forest / XGBoost |
| **C** | 定长序列（压缩） | 31 维压缩特征 | Random Forest / XGBoost |
| **D** | 定长序列（原始） | 25 维 × H 步时序 | LSTM / GRU |
| **E0** | 混合粒度（拼接） | ~89 维（单包+历史窗口+冷启动） | Random Forest / XGBoost |
| **E1** | 混合粒度（关系增强） | ~47 维（即时状态+历史基线+关系特征+阶段化冷启动） | Random Forest / XGBoost |

## 项目结构

```
.
├── main.py                          # 主入口，支持按阶段或按组运行
├── device_list.md                   # MAC 地址 → 设备名称映射
│
├── src/
│   ├── utils/
│   │   ├── config.py                # 全局配置（路径、特征列表、超参数）
│   │   └── logger.py                # 日志工具
│   ├── preprocessing/
│   │   ├── pcap_loader.py           # PCAP → 包级 CSV 解析
│   │   ├── packet_level_loader.py   # Group A 数据加载器
│   │   ├── window_loader.py         # Group B 数据加载器
│   │   ├── sequence_loader.py       # Group C/D 数据加载器
│   │   ├── group_e_loader.py        # Group E0 混合粒度加载器（baseline）
│   │   └── group_e1_loader.py       # Group E1 关系增强加载器
│   ├── features/
│   │   ├── window_features.py       # 窗口级统计特征计算
│   │   └── sequence_compressed_features.py  # 序列压缩特征计算
│   ├── models/
│   │   ├── classifier.py            # RF / XGBoost 分类器封装
│   │   └── rnn_classifier.py        # LSTM / GRU 训练器（含早停）
│   └── evaluation/
│       └── clf_metrics.py           # 评估指标 + 混淆矩阵 + 报告生成
│
├── experiments/
│   └── run_experiments.py           # 实验编排（5 阶段流水线）
│
├── data/
│   ├── 16-09-23.pcap                # UNSW-IoT 原始 PCAP（353 MB）
│   └── processed/                   # 预处理输出（自动生成）
│       ├── packet_level.csv         # 包级统一表征
│       ├── window_samples_*.csv     # Group B 窗口样本
│       ├── seq_compressed_*.csv     # Group C 压缩序列样本
│       ├── seq_raw_H*_L*/          # Group D 原始序列张量（.npy）
│       ├── group_e_samples_*.csv   # Group E0 混合粒度样本
│       └── group_e1_samples_*.csv  # Group E1 关系增强样本
│
└── results/
    ├── experiment_report.md         # 实验分析报告
    ├── experiments/                  # 各组实验结果
    │   ├── all_results.csv          # 汇总结果
    │   └── group_*/                 # 每组：metrics.json + per_class.csv + confusion.png
    ├── tables/                      # 汇总表格
    │   ├── sensitivity_*.csv        # 参数敏感性结果
    │   ├── feature_importance_*.csv # 特征重要性排名
    │   └── overall_summary.csv      # 总体汇总
    ├── figures/                      # 可视化图表
    │   ├── sensitivity_*.png        # 参数敏感性折线图
    │   └── feature_importance_*.png # 特征重要性柱状图
    ├── models/                      # Group D 训练好的模型权重
    └── logs/                        # 运行日志
```

## 环境依赖

- Python 3.10+
- dpkt（PCAP 解析）
- pandas, numpy
- scikit-learn
- xgboost
- torch（PyTorch，LSTM/GRU）
- seaborn, matplotlib（可视化）

```bash
pip install dpkt pandas numpy scikit-learn xgboost torch seaborn matplotlib
```

## 运行方式

### 完整流水线（5 个阶段）

```bash
python main.py --stage all
```

### 分阶段运行

```bash
python main.py --stage 1    # Stage 1: PCAP 解析 → packet_level.csv
python main.py --stage 2    # Stage 2: 四组主实验（A/B/C/D）
python main.py --stage 3    # Stage 3: 参数敏感性分析
python main.py --stage 4    # Stage 4: 特征重要性分析
python main.py --stage 5    # Stage 5: 结果汇总
```

### 单独运行某组实验

```bash
python main.py --group a --model rf          # Group A + Random Forest
python main.py --group b --model xgb --window 300  # Group B + XGBoost, W=300s
python main.py --group c --model rf --seq_len 20   # Group C + RF, H=20
python main.py --group d --model lstm --seq_len 20  # Group D + LSTM, H=20
python main.py --group e --model rf --sample_step 10 --hist_window 30   # Group E0
python main.py --group e1 --model xgb --sample_step 10 --hist_window 60 # Group E1
```

## 数据集

使用 UNSW-IoT Traces 数据集中的 `16-09-23.pcap`，包含智能家居环境中 31 种 IoT 设备的网络流量。经 MAC 地址映射和最小样本过滤（≥1000 包）后，保留 **18 种设备**、**875,359 个数据包**。

数据按时间顺序划分为训练集（70%）、验证集（15%）、测试集（15%）。

## 实验结果概要

| 组 | 分类器 | 准确率 | 宏-F1 | 加权-F1 |
|----|--------|--------|--------|---------|
| A | RF | 97.49% | 93.25% | 97.48% |
| A | XGBoost | 97.60% | 92.24% | 97.56% |
| B | RF | 97.56% | 95.32% | 97.31% |
| B | XGBoost | 97.82% | 95.40% | 97.73% |
| C | RF | 97.18% | 88.75% | 96.83% |
| C | XGBoost | 97.64% | 90.12% | 97.53% |
| D | LSTM | **99.34%** | **95.90%** | 99.33% |
| D | GRU | 99.31% | 95.14% | 99.32% |
| E0 | RF (S=5,W=60) | 95.13% | 91.54% | 95.16% |
| E1 | XGB (S=10,W=60) | **98.11%** | 92.78% | 98.12% |

详细分析见 [results/experiment_report.md](results/experiment_report.md)。
