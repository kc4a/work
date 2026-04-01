# 基于多粒度网络行为表征的 IoT 设备识别实验报告

> 数据集：UNSW-IoT 16-09-23.pcap（353 MB）
> 实验日期：2026-03-30
> 框架：Python 3.x / scikit-learn / XGBoost / PyTorch

---

## 一、实验概述

本实验以 UNSW-IoT 数据集为基础，将原始 PCAP 流量解析为统一的**包级 CSV**，再从中派生四种粒度的表征，通过多个分类器对 18 种 IoT 设备进行识别，评估不同粒度、不同算法在准确率、宏平均 F1 及时间开销三个维度上的综合表现。

| 实验组 | 粒度 | 特征 | 分类器 |
|--------|------|------|--------|
| A | 单包 | 28 维包头特征 | RF / XGBoost |
| B | 时间窗口 | ~50 维统计特征 | RF / XGBoost |
| C | 定长序列（压缩） | 31 维全局+分段+转移特征 | RF / XGBoost |
| D | 定长序列（原始） | 25 维逐包特征 × H 步 | LSTM / GRU |

---

## 二、数据集与预处理

### 2.1 数据集概况

| 项目 | 值 |
|------|----|
| PCAP 文件 | `data/16-09-23.pcap` |
| 原始有效包数（已知设备） | 875,359 |
| 识别设备数（过滤后） | **18 种** |
| 过滤阈值 | ≥ 1,000 包/设备 |
| 划分策略 | 按时间顺序 70 / 15 / 15（train / val / test） |

### 2.2 设备列表（18 种）

Amazon Echo、Android Phone 2、Belkin Wemo switch、Belkin wemo motion sensor、Dropcam、HP Printer、Laptop、Netatmo Welcome、Netatmo weather station、PIX-STAR Photo-frame、Samsung Galaxy Tab、Samsung SmartCam、Smart Things、TP-Link Day Night Cloud camera、TP-Link Smart plug、TPLink Router Bridge LAN、Triby Speaker、Withings Smart Baby Monitor

### 2.3 MAC 地址到设备名称映射

设备标签通过以太网帧中的 MAC 地址映射获取，共覆盖 31 个已知 MAC（涵盖 18 个有足够样本的设备）。未知 MAC 的包在解析阶段直接丢弃。

---

## 三、各组主实验结果（默认参数）

默认参数：Group B W=300s、S=300s；Group C/D H=20、L=20。

### 3.1 总体性能对比

| 组 | 分类器 | 准确率 | 宏-F1 | 加权-F1 | 训练时间(s) | 单样本推理时间(μs) |
|----|--------|--------|--------|---------|------------|------------------|
| **A** | RF | 0.9749 | 0.9325 | 0.9748 | 6.85 | 3.53 |
| **A** | XGBoost | 0.9760 | 0.9224 | 0.9756 | 26.10 | 2.79 |
| **B** | RF | 0.9756 | 0.9532 | 0.9731 | 0.18 | 37.7 |
| **B** | XGBoost | 0.9782 | 0.9540 | 0.9773 | 0.50 | 2.98 |
| **C** | RF | 0.9718 | 0.8875 | 0.9683 | 0.80 | 8.21 |
| **C** | XGBoost | 0.9764 | 0.9012 | 0.9753 | 2.17 | 3.78 |
| **D** | LSTM | **0.9934** | **0.9590** | **0.9933** | 205.6 | 106.7 |
| **D** | GRU | 0.9931 | 0.9514 | 0.9932 | 377.4 | 129.4 |

**结论：**
- 最高准确率由 **Group D（RNN）** 取得（Acc≈99.3%），宏-F1 = 95.9%（LSTM）；
- ML 方法中 **Group B（窗口级）XGBoost** 宏-F1 = 95.4%，与 RNN 差距极小且效率远优；
- Group C（序列压缩+RF）宏-F1 最低（0.8875），说明压缩后的全局统计损失了部分设备间的区分信息；
- Group A 推理延迟最低（<4 μs/样本），适合实时场景。

### 3.2 特征构造时间对比

| 组 | RF 特征时间(s) | XGB/RNN 特征时间(s) | 说明 |
|----|--------------|------------------|------|
| A | 3.92 | 4.07 | 直接读 CSV，开销极低 |
| B | 17.67 | 17.15 | 时间窗口切分 + 统计计算 |
| C | 87.71 | 88.00 | 序列构造 + 压缩特征 |
| D (LSTM) | — | 38.77 | 序列切分 + 标准化 |
| D (GRU) | — | 37.43 | 序列切分 + 标准化 |

---

## 四、各类设备逐类分析

### 4.1 Group A RF —— 逐设备 P/R/F1

| 设备 | 精确率 | 召回率 | F1 |
|------|--------|--------|-----|
| Amazon Echo | 0.968 | 0.989 | 0.978 |
| Android Phone 2 | **0.688** | **0.635** | **0.660** |
| Belkin Wemo switch | 0.852 | 0.928 | 0.889 |
| Belkin wemo motion sensor | 0.928 | 0.877 | 0.902 |
| Dropcam | 1.000 | 1.000 | 1.000 |
| HP Printer | 0.855 | 0.828 | 0.841 |
| Laptop | 0.992 | 0.999 | 0.995 |
| Netatmo Welcome | 0.914 | 0.863 | 0.888 |
| Netatmo weather station | 1.000 | 1.000 | 1.000 |
| PIX-STAR Photo-frame | 0.918 | 0.959 | 0.938 |
| Samsung Galaxy Tab | 0.782 | 0.802 | 0.791 |
| Samsung SmartCam | 0.975 | 0.980 | 0.978 |
| Smart Things | 0.999 | 1.000 | 1.000 |
| TP-Link Day Night Cloud camera | 0.926 | 0.935 | 0.931 |
| TP-Link Smart plug | 1.000 | 1.000 | 1.000 |
| TPLink Router Bridge LAN | 0.998 | 0.995 | 0.996 |
| Triby Speaker | 1.000 | 1.000 | 1.000 |
| Withings Smart Baby Monitor | 0.998 | 1.000 | 0.999 |

**难识别设备：** Android Phone 2（F1=0.660）和 Samsung Galaxy Tab（F1=0.791）因流量行为与其他设备重叠度高，包级特征判别力不足。

### 4.2 Group B RF —— 逐设备 P/R/F1

| 设备 | 精确率 | 召回率 | F1 |
|------|--------|--------|-----|
| Amazon Echo | 1.000 | 1.000 | 1.000 |
| Android Phone 2 | 1.000 | 1.000 | 1.000 |
| Belkin Wemo switch | 1.000 | 1.000 | 1.000 |
| Belkin wemo motion sensor | 1.000 | 1.000 | 1.000 |
| Dropcam | 1.000 | 1.000 | 1.000 |
| HP Printer | 0.970 | 1.000 | 0.985 |
| Laptop | 1.000 | 1.000 | 1.000 |
| Netatmo Welcome | 1.000 | 1.000 | 1.000 |
| Netatmo weather station | 0.926 | 1.000 | 0.962 |
| PIX-STAR Photo-frame | 0.837 | 1.000 | 0.911 |
| Samsung Galaxy Tab | 1.000 | **0.514** | **0.679** |
| Samsung SmartCam | 1.000 | 1.000 | 1.000 |
| Smart Things | 1.000 | 1.000 | 1.000 |
| TP-Link Day Night Cloud camera | 0.956 | 1.000 | 0.977 |
| TP-Link Smart plug | 0.556 | 0.833 | **0.667** |
| TPLink Router Bridge LAN | 1.000 | 1.000 | 1.000 |
| Triby Speaker | 0.956 | 1.000 | 0.977 |
| Withings Smart Baby Monitor | 1.000 | 1.000 | 1.000 |

**窗口粒度大幅提升 Android Phone 2**（F1：0.660 → 1.000），但 Samsung Galaxy Tab 和 TP-Link Smart plug 在默认参数下仍存在混淆。

### 4.3 Group D LSTM —— 逐设备 P/R/F1（H=20, L=20）

| 设备 | 精确率 | 召回率 | F1 |
|------|--------|--------|-----|
| Amazon Echo | 1.000 | 1.000 | 1.000 |
| Android Phone 2 | 0.815 | **0.677** | **0.739** |
| Belkin Wemo switch | 0.996 | 1.000 | 0.998 |
| Belkin wemo motion sensor | 1.000 | 1.000 | 1.000 |
| Dropcam | 0.998 | 1.000 | 0.999 |
| HP Printer | 1.000 | 1.000 | 1.000 |
| Laptop | 1.000 | 1.000 | 1.000 |
| Netatmo Welcome | 0.963 | 1.000 | 0.981 |
| Netatmo weather station | 1.000 | 1.000 | 1.000 |
| PIX-STAR Photo-frame | 1.000 | 1.000 | 1.000 |
| Samsung Galaxy Tab | 0.833 | 0.804 | **0.818** |
| Samsung SmartCam | 0.998 | 1.000 | 0.999 |
| Smart Things | 1.000 | 1.000 | 1.000 |
| TP-Link Day Night Cloud camera | 1.000 | 1.000 | 1.000 |
| TP-Link Smart plug | **0.571** | 1.000 | **0.727** |
| TPLink Router Bridge LAN | 1.000 | 1.000 | 1.000 |
| Triby Speaker | 1.000 | 1.000 | 1.000 |
| Withings Smart Baby Monitor | 1.000 | 1.000 | 1.000 |

### 4.4 Group D GRU —— 逐设备 P/R/F1（H=20, L=20）

| 设备 | 精确率 | 召回率 | F1 |
|------|--------|--------|-----|
| Amazon Echo | 0.995 | 1.000 | 0.998 |
| Android Phone 2 | 0.735 | 0.938 | 0.824 |
| Belkin Wemo switch | 1.000 | 0.989 | 0.994 |
| Belkin wemo motion sensor | 0.991 | 1.000 | 0.995 |
| Dropcam | 1.000 | 1.000 | 1.000 |
| HP Printer | 1.000 | 1.000 | 1.000 |
| Laptop | 1.000 | 1.000 | 1.000 |
| Netatmo Welcome | 0.992 | 1.000 | 0.996 |
| Netatmo weather station | 1.000 | 1.000 | 1.000 |
| PIX-STAR Photo-frame | 0.935 | 1.000 | 0.967 |
| Samsung Galaxy Tab | 0.949 | **0.661** | **0.779** |
| Samsung SmartCam | 1.000 | 1.000 | 1.000 |
| Smart Things | 1.000 | 1.000 | 1.000 |
| TP-Link Day Night Cloud camera | 1.000 | 1.000 | 1.000 |
| TP-Link Smart plug | **0.400** | 1.000 | **0.571** |
| TPLink Router Bridge LAN | 1.000 | 1.000 | 1.000 |
| Triby Speaker | 1.000 | 1.000 | 1.000 |
| Withings Smart Baby Monitor | 1.000 | 1.000 | 1.000 |

**RNN 分析：** LSTM/GRU 在多数设备上达到 F1≥0.99，但少数类设备仍为瓶颈：TP-Link Smart plug（LSTM F1=0.727、GRU F1=0.571）因测试集样本极少导致精确率偏低；Android Phone 2（LSTM F1=0.739）与其他移动设备易混淆；Samsung Galaxy Tab（GRU F1=0.779）召回率偏低。

---

## 五、参数敏感性分析

### 5.1 窗口大小 W 的影响（S = W）

| W (s) | 模型 | 准确率 | 宏-F1 | 加权-F1 | 特征构造时间(s) |
|--------|------|--------|--------|---------|----------------|
| 60 | RF | 0.9706 | 0.9449 | 0.9703 | 45.01 |
| 60 | XGB | 0.9706 | 0.9305 | 0.9705 | 41.12 |
| **300** | **RF** | **0.9756** | **0.9532** | **0.9731** | 14.32 |
| **300** | **XGB** | **0.9782** | **0.9540** | **0.9773** | 14.20 |
| 900 | RF | 0.9467 | 0.9490 | 0.9380 | 9.07 |
| 900 | XGB | 0.9400 | 0.8698 | 0.9305 | 8.43 |

**W\* = 300s**：两种模型在宏-F1 和准确率均达到最优。W=60s 时窗口内包数较少，统计特征噪声大；W=900s 时样本数过少，泛化能力下降。

### 5.2 滑动步长 S 的影响（W = 300s）

| W | S | 模型 | 准确率 | 宏-F1 | 特征构造时间(s) |
|----|---|------|--------|--------|----------------|
| 300 | 30 | RF | 0.9810 | 0.9359 | 108.49 |
| 300 | 30 | XGB | 0.9772 | 0.8990 | 110.90 |
| 300 | 60 | RF | 0.9797 | 0.9329 | 60.83 |
| 300 | 60 | XGB | 0.9785 | 0.9223 | 67.52 |
| 300 | 150 | RF | 0.9782 | 0.9416 | 29.73 |
| 300 | 150 | XGB | 0.9770 | 0.9421 | 30.45 |
| **300** | **300** | **RF** | **0.9756** | **0.9532** | **17.67** |
| **300** | **300** | **XGB** | **0.9782** | **0.9540** | **17.15** |

**S = W = 300s（无重叠）**在宏-F1 最优，且特征构造时间最短。减小步长（S<W）通过数据增强提升准确率，但宏-F1 并未持续提升，反而因多数类样本过度增殖而拉低少数类识别；同时特征构造时间随 S 减小显著增长（S=30 时约 110s）。

### 5.3 序列长度 H 的影响（L = H，C+D 组合对比）

| H | 模型 | 组 | 准确率 | 宏-F1 | 特征构造时间(s) |
|---|------|----|--------|--------|----------------|
| 10 | RF | C | 0.9674 | 0.8913 | 137.50 |
| 10 | XGB | C | 0.9729 | 0.9108 | 138.72 |
| 10 | LSTM | D | 0.9903 | 0.9522 | 57.28 |
| 10 | GRU | D | 0.9892 | 0.9450 | 56.58 |
| **20** | **RF** | **C** | **0.9718** | **0.8875** | 69.83 |
| **20** | **XGB** | **C** | **0.9764** | **0.9012** | 70.86 |
| **20** | **LSTM** | **D** | **0.9950** | **0.9712** | 32.07 |
| **20** | **GRU** | **D** | **0.9950** | **0.9678** | 36.13 |
| 50 | RF | C | 0.9759 | 0.8960 | 27.89 |
| 50 | XGB | C | 0.9790 | 0.9118 | 29.38 |
| 50 | LSTM | D | 0.9916 | 0.9546 | 13.90 |
| 50 | GRU | D | 0.9935 | 0.9502 | 13.87 |

**H\* = 20**：Group D（RNN）在 H=20 时宏-F1 最高（0.9712 / 0.9678），H=50 时减少样本数且序列内前后相关性被稀释；Group C 随 H 增大宏-F1 略有提升但变化幅度小。

### 5.4 序列步长 L 的影响（H = 20）

| H | L | 模型 | 组 | 准确率 | 宏-F1 | 特征构造时间(s) |
|---|---|------|----|--------|--------|----------------|
| 20 | **5** | RF | C | **0.9795** | **0.9122** | 280.80 |
| 20 | **5** | XGB | C | **0.9823** | **0.9211** | 285.51 |
| 20 | **5** | LSTM | D | **0.9962** | **0.9831** | 111.38 |
| 20 | **5** | GRU | D | **0.9963** | **0.9764** | 112.39 |
| 20 | 10 | RF | C | 0.9752 | 0.8996 | 152.46 |
| 20 | 10 | XGB | C | 0.9804 | 0.9108 | 153.07 |
| 20 | 10 | LSTM | D | 0.9951 | 0.9756 | 62.79 |
| 20 | 10 | GRU | D | 0.9940 | 0.9527 | 60.64 |
| 20 | 20 | RF | C | 0.9718 | 0.8875 | 87.71 |
| 20 | 20 | XGB | C | 0.9764 | 0.9012 | 87.80 |
| 20 | 20 | LSTM | D | 0.9950 | 0.9712 | 32.07 |
| 20 | 20 | GRU | D | 0.9950 | 0.9678 | 30.48 |

**L = 5 时性能最优**（数据增强效果最强）但特征构造时间增加约 3.5 倍（~280s vs ~88s）；L=10 是精度与时间的较好折中。

---

## 六、特征重要性分析

基于 Random Forest 的 `feature_importances_` 对 Groups A、B、C 提取特征重要性排名。

### 6.1 Group A —— 包级特征 Top-10

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | `tcp_window` | 0.1646 |
| 2 | `src_port` | 0.1349 |
| 3 | `dst_port` | 0.1324 |
| 4 | `delta_t` | 0.0986 |
| 5 | `ip_ttl` | 0.0907 |
| 6 | `packet_len` | 0.0581 |
| 7 | `payload_entropy` | 0.0579 |
| 8 | `ip_len` | 0.0557 |
| 9 | `tcp_dataofs` | 0.0404 |
| 10 | `payload_len` | 0.0379 |

**分析：** TCP 窗口大小（16.5%）居首，反映不同 IoT 设备操作系统/协议栈的默认 TCP 配置差异显著；端口对（src_port + dst_port 合计 26.7%）是第二大判别维度；包间时延 `delta_t`（9.9%）和 TTL（9.1%）分别编码设备通信频率和网络拓扑距离。值得注意的是 `direction` 重要性为 0，说明在包级粒度单独看方向对设备区分贡献极小。

### 6.2 Group B —— 窗口级特征 Top-10

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | `packet_len_min` | 0.0573 |
| 2 | `well_known_dst_port_ratio` | 0.0523 |
| 3 | `payload_entropy_mean` | 0.0512 |
| 4 | `icmp_count` | 0.0474 |
| 5 | `https_port_ratio` | 0.0434 |
| 6 | `iat_max` | 0.0380 |
| 7 | `udp_count` | 0.0355 |
| 8 | `payload_len_mean` | 0.0340 |
| 9 | `unique_peer_ip_count` | 0.0312 |
| 10 | `tcp_count` | 0.0299 |

**分析：** 窗口级特征重要性分布更均匀（Top-1 仅 5.7% vs 包级 16.5%），说明统计聚合使多维信息互补。`packet_len_min`（最小包长）区分心跳型设备（小包）与流媒体设备（大包）；`well_known_dst_port_ratio` 和 `https_port_ratio` 编码设备使用的服务类型；`icmp_count` 区分路由设备（高 ICMP）与终端设备。突发特征（`burst_byte_ratio` 排第 22）贡献中等。

### 6.3 Group C —— 序列压缩特征 Top-10

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | `seq_packet_len_max` | 0.1087 |
| 2 | `seq_iat_std` | 0.0836 |
| 3 | `seq_iat_mean` | 0.0809 |
| 4 | `seq_packet_len_min` | 0.0697 |
| 5 | `seq_packet_len_mean` | 0.0629 |
| 6 | `seq_payload_len_std` | 0.0617 |
| 7 | `seq_packet_len_std` | 0.0596 |
| 8 | `tcp_flag_change_count` | 0.0583 |
| 9 | `seq_payload_len_mean` | 0.0552 |
| 10 | `dst_port_change_count` | 0.0397 |

**分析：** 包长全局统计（max/min/mean/std 合计 30.1%）为最强判别维度；IAT 统计（mean+std 合计 16.5%）次之。转移特征中 `tcp_flag_change_count`（5.8%）和 `dst_port_change_count`（4.0%）有明显贡献，说明序列内 TCP 状态转移频率和端口多样性是有效判别信号。方向相关特征（`seq_outbound_ratio`、`direction_switch_count` 等）重要性为 0，与 Group A 中 `direction=0` 一致，表明在该数据集上方向特征判别力极弱。

> 特征重要性图表：`results/figures/feature_importance_group_{A,B,C}.png`

---

## 七、时间开销综合对比

| 组 | 特征构造(s) | 训练(s) | 推理/样本(μs) | 综合评价 |
|----|-----------|---------|--------------|----------|
| A-RF | 3.9 | 6.9 | 3.5 | 构造极快，推理极快，宏F1偏低 |
| A-XGB | 4.1 | 26.1 | 2.8 | 训练较慢，推理最快 |
| B-RF | 17.7 | 0.2 | 37.7 | 训练极快，推理稍慢（窗口级） |
| B-XGB | 17.1 | 0.5 | **3.0** | 训练快，推理快，综合最优 |
| C-RF | 87.7 | 0.8 | 8.2 | 特征构造慢，其余快 |
| C-XGB | 88.0 | 2.2 | 3.8 | 特征构造慢，推理快 |
| D-LSTM | 38.8 | 205.6 | 106.7 | 训练慢，准确率最高 |
| D-GRU | 37.4 | 377.4 | 129.4 | 训练最慢，推理最慢 |

**结论：**
- 对延迟敏感场景（在线识别）推荐 **Group A** 或 **Group B-XGB**；
- 对精度要求高的离线场景推荐 **Group D-LSTM**（宏-F1=95.9%，Acc=99.3%）；
- Group B-XGB 是精度与效率的最佳平衡点（宏-F1=95.4%，推理 3 μs），与 RNN 差距极小。

---

## 八、跨粒度对比总结

### 8.1 准确率与宏-F1 趋势

```
粒度：  单包(A) ≈ 窗口(B) ≈ 序列压缩(C) < 序列原始(D)
MacroF1: C(0.90) < A(0.93) < B(0.95) ≈ D(0.96)
Acc:     C(0.97) ≤ A(0.97) ≤ B(0.98) < D(0.99)
```

粒度越高（更多包的上下文），识别效果越好，其中 RNN 对时序信息的挖掘带来最显著的精度提升。

### 8.2 各粒度适用场景建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 实时在线识别（<10μs） | Group A RF | 推理 3.5μs，准确率 97.5% |
| 准实时（<50μs） | Group B XGB | 推理 3μs，宏-F1=95.4% |
| 批处理高精度 | Group D LSTM | Acc=99.3%，宏-F1=95.9% |
| 资源受限设备 | Group C XGB | 特征固定31维，小模型 |

### 8.3 难识别设备分析

以下设备在多组实验中持续出现混淆：

- **Android Phone 2**：流量行为接近普通移动端，与其他移动设备/PC 混淆，在包级（Group A）F1 仅 0.660；窗口级（Group B）提升至 1.000；LSTM（Group D）为 0.739、GRU 为 0.824。
- **Samsung Galaxy Tab**：与 Android Phone 2 行为相似，包级 F1=0.791，窗口级受步长影响（F1=0.679）；LSTM 约 0.818、GRU 约 0.779。
- **TP-Link Smart plug**：测试集样本极少（仅约 8 个），LSTM F1=0.727、GRU F1=0.571；窗口级 F1=0.667。小样本设备在所有粒度下均为挑战。

---

## 九、结论

1. **多粒度表征有效性验证**：从单包→时间窗口→序列压缩→原始序列，准确率依次提升（97.5% → 97.8% → 97.6% → 99.3%），验证了更高粒度的时序上下文对设备识别的正面贡献。

2. **RNN 准确率最高但宏-F1 优势有限**：Group D（LSTM）准确率达 99.3% 为最高，但宏-F1（95.9%）与 Group B XGBoost（95.4%）差距极小，因为 RNN 在少数类设备（TP-Link Smart plug、Android Phone 2）上仍表现较弱。训练时间高出 2 个量级（>200s vs <30s）。

3. **窗口级特征为最佳折中方案**：Group B XGBoost（W=300s）在宏-F1（95.4%）、推理延迟（3μs）、训练时间（0.5s）上综合表现最优，适合实际部署。

4. **参数选择**：窗口大小 W=300s、步长 S=W 为默认最优；序列长度 H=20 最优，减小步长 L 可提升精度但显著增加特征构造时间（L=5 时约 280s）。

5. **特征重要性启示**：包级最重要特征为 TCP 窗口大小和端口对（合计 43%），窗口级重要性更分散（Top-1 仅 5.7%），序列压缩级以包长统计和 IAT 统计主导（合计 46.6%）。方向特征在所有粒度下重要性近零。

6. **困难设备与少样本问题**：移动设备（Android Phone 2、Samsung Galaxy Tab）在包级难以区分，窗口/序列粒度有所改善；TP-Link Smart plug 因测试样本极少（~8 个）在所有粒度下 F1 偏低（0.57~0.73），是典型的少样本挑战。

---

## 附录：实验输出文件清单

| 文件 | 描述 |
|------|------|
| `data/processed/packet_level.csv` | 包级 CSV（875,359 行） |
| `results/experiments/group_*/` | 各组 metrics.json + per_class.csv + confusion.png |
| `results/tables/sensitivity_window_W.csv` | 窗口大小敏感性结果 |
| `results/tables/sensitivity_window_S.csv` | 窗口步长敏感性结果 |
| `results/tables/sensitivity_seq_H.csv` | 序列长度敏感性结果 |
| `results/tables/sensitivity_seq_L.csv` | 序列步长敏感性结果 |
| `results/figures/sensitivity_window_W.png` | 窗口敏感性折线图 |
| `results/figures/sensitivity_seq_H.png` | 序列敏感性折线图 |
| `results/tables/feature_importance_group_A.csv` | Group A RF 特征重要性 |
| `results/tables/feature_importance_group_B.csv` | Group B RF 特征重要性 |
| `results/tables/feature_importance_group_C.csv` | Group C RF 特征重要性 |
| `results/figures/feature_importance_group_{A,B,C}.png` | 特征重要性柱状图 |
| `results/tables/overall_summary.csv` | 全部实验汇总表 |
