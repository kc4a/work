# UNSW-IoTraffic 子集跨数据集验证报告

## 1. 实验目的

本实验使用 `data/pcaps` 中下载的 UNSW-IoTraffic 设备级 PCAP，对当前论文框架进行一轮新的跨数据集验证。

与此前使用 `IoT_2023-07-24.pcap` 的外部跨域测试不同，这一轮更适合被定义为：

- 同设备集合下的跨时间验证
- 跨采集批次验证
- 中等强度跨域验证

原因在于：该数据集与当前主数据集存在较高的设备重叠，且大量设备的 MAC 地址完全一致，因此它不是强异构家庭环境，而是同体系下不同时间段的设备流量。

## 2. 数据构造方式

### 2.1 数据来源

- 原始训练集：`data/16-09-23.pcap`
- 新测试集：`data/pcaps/*.pcap`

### 2.2 设备重叠

本轮实验共选取 14 个与当前标签体系直接重叠的设备：

- Amazon Echo
- Belkin Wemo switch
- Belkin wemo motion sensor
- Dropcam
- HP Printer
- Netatmo Welcome
- Netatmo weather station
- PIX-STAR Photo-frame
- Samsung SmartCam
- Smart Things
- TP-Link Day Night Cloud camera
- TP-Link Smart plug
- Triby Speaker
- Withings Smart Baby Monitor

### 2.3 子集抽取策略

考虑到每个设备 PCAP 体量较大，本实验未使用全量流量，而是对每个设备 PCAP 仅截取前一段连续流量：

- 每个设备保留前 `20,000` 个可识别包
- 合并后生成测试 CSV：
  - `data/processed/unsw_iotraffic_subset_packet_level.csv`

最终得到：

- 14 个设备
- 280,000 条包级样本

### 2.4 运行组别

本轮实验覆盖以下组别：

- A：单包级
- B：窗口级
- C：序列压缩级
- D：原始序列级
- E0：混合粒度 baseline
- E1：关系增强混合粒度

## 3. 总体结果

汇总结果见：

- `results/tables/unsw_iotraffic_subset_cross_summary.csv`

| 组 | 模型 | Accuracy | Macro-F1 | Weighted-F1 |
|----|------|----------|----------|-------------|
| A | RF | 0.5921 | 0.6183 | 0.6183 |
| A | XGB | 0.5735 | 0.5928 | 0.5928 |
| B | RF | **0.8603** | **0.8280** | **0.8636** |
| B | XGB | 0.7425 | 0.6956 | 0.7100 |
| C | RF | 0.5230 | 0.4196 | 0.5395 |
| C | XGB | 0.3637 | 0.2766 | 0.3556 |
| D | LSTM | 0.6087 | 0.4693 | 0.6033 |
| D | GRU | 0.6041 | 0.4919 | 0.5973 |
| E0 | RF | 0.5904 | 0.7656 | 0.6200 |
| E0 | XGB | 0.5854 | 0.7698 | 0.6060 |
| E1 | RF | 0.8588 | 0.7884 | 0.8628 |
| E1 | XGB | 0.6686 | 0.6214 | 0.6603 |

## 4. 总体结论

### 4.1 最优方案

本轮 UNSW-IoTraffic 子集跨数据集验证中，表现最好的方案是：

- `B-RF`：Accuracy = `0.8603`，Macro-F1 = `0.8280`
- `E1-RF`：Accuracy = `0.8588`，Macro-F1 = `0.7884`

这说明在“同设备、跨时间/跨采集批次”的设置下：

- 窗口级统计特征最稳健
- 关系增强的混合粒度方案也具有较强稳定性

### 4.2 中等表现

- A 组整体处于中等水平，说明单包特征可以保留一部分可迁移性，但稳定性明显不如窗口级。
- E0 的 Accuracy 不高，但 Macro-F1 相对较高，说明它在类别均衡性上保留了一定优势。

### 4.3 最弱方案

表现最差的是 C 组：

- `C-RF`：Macro-F1 = `0.4196`
- `C-XGB`：Macro-F1 = `0.2766`

这说明序列压缩特征对训练批次中的统计结构依赖较强，一旦换到新的采集阶段，压缩后的全局统计模式最容易失效。

### 4.4 D 组的含义

- D-LSTM 与 D-GRU 的 Accuracy 都在 `0.60` 左右
- 明显低于 B-RF 和 E1-RF

这说明原始序列级 RNN 在这类“同设备跨时间”场景里并没有体现出额外优势，反而可能学到了较多训练批次内的时序细节模式。

## 5. 逐组分析

### 5.1 A 组：单包级

- A-RF 和 A-XGB 均保持在 `0.57-0.59` 的准确率
- 说明单包级特征具备一定迁移能力
- 但它们对协议栈配置、端口分布、TTL 等域相关特征依赖较强，因此稳定性有限

### 5.2 B 组：窗口级

- B-RF 在本轮实验中取得最佳总体表现
- 这说明聚合统计比单包瞬时特征更能跨时间保持稳定
- 特别是在同设备、不同采集批次的设定下，窗口行为模式是最稳妥的判别依据

### 5.3 C 组：序列压缩级

- C 组显著弱于其他组
- 这说明压缩特征虽然在同域测试中可以工作，但在新批次下容易丢失关键差异
- 换言之，它压缩掉的不只是噪声，也压缩掉了新的时间段里仍然有用的局部判别信息

### 5.4 D 组：原始序列级

- D 组比 C 组强，但仍明显弱于 B
- 说明 RNN 学到的顺序模式并没有天然比窗口统计更稳
- 在新的采集批次里，顺序结构可能发生偏移，从而削弱模型泛化

### 5.5 E0 与 E1

- E0 的 Macro-F1 较高，但 Accuracy 一般
- E1-RF 的 Accuracy 接近 B-RF
- 这表明“当前包 + 历史基线 + 关系增强”在同设备跨时间场景中具有较好适应性

不过：

- E1-XGB 明显不如 E1-RF
- 说明 E1 结构本身有效，但树模型类型不同，对关系增强特征的利用方式有差异

## 6. 逐设备现象

### 6.1 A-RF

从 `results/experiments/unsw_iotraffic_subset_group_a_rf/cross_dataset_group_a_rf_per_class.csv` 看：

- `Dropcam`、`Netatmo weather station`、`Smart Things`、`Withings Smart Baby Monitor` 精确率很高，但召回率大多只有约 `0.49-0.51`
- `Belkin wemo motion sensor` 最难，F1 仅约 `0.253`
- `Belkin Wemo switch`、`Netatmo Welcome`、`HP Printer` 也存在明显下降

这说明单包特征在跨批次下保留了“能识别是谁”的部分强指纹，但不能稳定召回所有样本。

### 6.2 B-RF

从 `results/experiments/unsw_iotraffic_subset_group_b_rf_W300_S300/cross_dataset_group_b_rf_W300_S300_per_class.csv` 看：

- 大多数设备 F1 处于较高水平
- `Amazon Echo`、`Dropcam`、`HP Printer`、`Netatmo weather station`、`Smart Things`、`TP-Link Day Night Cloud camera`、`Withings Smart Baby Monitor` 都相对稳定
- 相对较弱的是 `Belkin Wemo switch`、`Belkin wemo motion sensor`、`Netatmo Welcome`

总体上，B-RF 兼顾了高准确率和较好的类别平衡性。

### 6.3 E1-RF

从 `results/experiments/unsw_iotraffic_subset_group_e1_rf_S10_W30/cross_dataset_group_e1_rf_S10_W30_per_class.csv` 看：

- `Dropcam`、`Netatmo weather station`、`TP-Link Day Night Cloud camera`、`Triby Speaker` 等设备表现稳定
- `Belkin Wemo switch` 和 `Belkin wemo motion sensor` 仍然是主要薄弱类
- `PIX-STAR Photo-frame` 和 `Samsung SmartCam` 也存在明显回落

说明 E1 对部分设备的“关系增强”是有效的，但并没有像窗口级统计那样对所有类都足够稳定。

### 6.4 D-GRU

从 `results/experiments/unsw_iotraffic_subset_group_d_gru_H50_L10/cross_dataset_group_d_gru_H50_L10_per_class.csv` 看：

- `Netatmo weather station`、`Smart Things`、`Triby Speaker` 仍然很强
- `HP Printer`、`Netatmo Welcome`、`TP-Link Smart plug` 中等
- `Amazon Echo`、`Belkin wemo motion sensor` 较弱
- `Dropcam`、`PIX-STAR Photo-frame` 等设备在 D 组中会出现极低召回甚至接近失效

这再次说明，RNN 在这次跨批次验证中对部分设备的时序模式学得过于依赖训练批次。

## 7. 与此前 IoT_2023 跨数据集实验的差异

这次结果和此前 `IoT_2023-07-24.pcap` 的外部跨域实验差异很大。

此前外部跨域实验的核心结论是：

- 所有模型整体大幅下降
- 高粒度模型并不更稳
- 泛化能力整体很弱

而这次 UNSW-IoTraffic 子集实验表明：

- 窗口级和关系增强方案可以达到 `0.85+` 的 Accuracy
- 说明同设备、同体系、跨时间的泛化并没有那么难
- 也说明此前 IoT_2023 的失败，很大程度上来自“设备代际 + 网络环境 + 采集场景”同时变化的强域偏移

因此论文中应将两类跨数据集验证明确区分：

- `UNSW-IoTraffic 子集`：中等强度跨域，强调跨时间/跨批次泛化
- `IoT_2023 外部数据`：强异构跨域，强调跨环境/跨代际泛化

## 8. 论文可直接使用的总结

本轮基于 UNSW-IoTraffic 设备级 PCAP 子集的跨数据集验证表明，在与训练集高度重叠的设备集合、但不同采集批次的场景下，窗口级统计特征和关系增强混合粒度特征具有较好的时间泛化能力。其中，B-RF 取得了最高的总体性能（Accuracy = 86.0%，Macro-F1 = 82.8%），E1-RF 紧随其后（Accuracy = 85.9%，Macro-F1 = 78.8%）。相比之下，序列压缩特征（C）和原始序列 RNN（D）并未表现出更强稳定性，说明更高粒度的时序建模并不一定在跨批次条件下天然更优。这一结果与此前在 IoT_2023 外部数据上的强跨域失败形成对照，说明当前方法对“同设备跨时间”仍有较好适应性，但对“跨环境、跨代际”的泛化能力仍然有限。
