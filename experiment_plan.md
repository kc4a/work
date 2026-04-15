IoT 设备识别毕业设计实验计划

_主线：多粒度行为表示比较 + 早期识别 / 最小上下文识别能力 + 性能-成本权衡_

**项目一句话概括：** 以 packet-level CSV 为统一底层表示，比较单包级、窗口级、序列压缩级和原始序列级四种输入粒度在 IoT 设备识别中的表现，并分析不同上下文长度下的早期识别能力、类别级表现与计算成本。

# 一、项目目标

本项目关注的问题是：在加密流量场景下，IoT 设备识别究竟需要多大的行为上下文，以及不同粒度的网络表示在识别效果和计算成本之间如何权衡。

项目只做设备识别，不做流量识别。统一底层表示来自原始 pcap 提取后的 packet-level CSV，一行代表一个包。

项目的核心不仅是比较谁更准，还要回答：单包是否足够、局部时间行为是否更稳、弱顺序摘要是否已经可用、显式时序建模是否值得其成本。

# 二、总体实验结构

Group A：Packet-level + ML。样本为一个包，模型为 Random Forest 和 XGBoost，作用是最基础的最小上下文 baseline。

Group B：Window-level + ML。样本为一个时间窗口，模型为 Random Forest 和 XGBoost，作用是比较局部时间行为统计是否优于单包。

Group C：Sequence-compressed + ML。样本为一个定长包序列，但输入是从序列压缩得到的固定长度特征向量，模型为 Random Forest 和 XGBoost，作用是比较弱顺序摘要是否已经足够。

Group D：Sequence-level + RNN。样本为一个定长包序列，输入为原始逐包序列，模型为 LSTM 和 GRU，作用是比较显式顺序建模是否还能进一步提升。

# 三、统一底层数据：packet-level CSV

基础规则：一行等于一个包；每个数据集各自处理；输出 CSV 的列名和字段语义统一。

## 推荐保留字段

| 类别            | 字段                                                                          |
| --------------- | ----------------------------------------------------------------------------- |
| 标识字段        | dataset_name, capture_id, packet_id, ts                                       |
| 地址与协议字段  | src_ip, dst_ip, src_port, dst_port, ip_proto, ip_version, eth_type            |
| IP 层字段       | ip_len, ip_ttl, ip_tos, ip_flags, ip_ihl, ip_df                               |
| TCP 层字段      | tcp_syn, tcp_ack, tcp_fin, tcp_rst, tcp_psh, tcp_urg, tcp_window, tcp_dataofs |
| UDP / ICMP 字段 | udp_len, icmp_type, icmp_code                                                 |
| 负载与方向字段  | packet_len, payload_len, has_payload, payload_entropy, direction              |
| 上下文字段      | delta_t                                                                       |
| 标签字段        | device_id, split                                                              |

注意：不使用ip，mac地址作为训练特

# 四、实验组 A：单包级 + ML

## 研究问题

单个包本身包含多少设备身份信息；这组实验对应最早期识别能力。

## 样本定义

• 一个样本 = 一个包

• 标签 = device_id

## 模型

• Random Forest

• XGBoost

## 输入特征

### 包大小与负载特征

**packet_len：**整个包长度，用于描述最基础的发包规模模式。

**payload_len：**有效载荷长度，比总包长更接近实际业务内容。

**has_payload：**是否携带载荷，用于区分控制包与数据包。

**payload_entropy：**载荷熵，衡量内容随机性，辅助区分明文、压缩与加密负载。

### 时间特征

**delta_t：**当前包与前一个包之间的时间间隔，反映局部发包节奏。

### 协议与端口特征

**ip_proto：**协议类型，例如 TCP、UDP、ICMP。

**src_port：**源端口，描述发起端端口行为。

**dst_port：**目的端口，反映常访问服务类型。

**eth_type：**链路层协议类型，补充基础协议环境信息。

**ip_version：**IPv4 或 IPv6，反映不同设备在 IP 版本使用上的差异。

### IP 层特征

**ip_len：**IP 总长度。

**ip_ttl：**TTL，反映协议栈差异。

**ip_tos：**Type of Service / DSCP。

**ip_flags：**IP 标志位，反映分片及协议栈行为。

**ip_ihl：**IP 头长度。

**ip_df：**Don't Fragment 标志。

### TCP 层特征

**tcp_syn：**是否为 SYN 包，反映建连行为。

**tcp_ack：**是否为 ACK 包，反映确认行为。

**tcp_fin：**是否为 FIN 包，反映连接终止。

**tcp_rst：**是否为 RST 包，反映异常重置或拒绝。

**tcp_psh：**是否为 PSH 包，反映数据推送行为。

**tcp_urg：**是否为 URG 包。

**tcp_window：**TCP 窗口大小，反映 TCP 栈行为。

**tcp_dataofs：**TCP 头长度，间接反映选项与头部结构。

### UDP / ICMP 与方向特征

**udp_len：**UDP 长度字段。

**icmp_type：**ICMP 类型。

**icmp_code：**ICMP 代码。

**direction：**入站 / 出站方向，反映设备主动与被动通信倾向。

# 五、实验组 B：窗口级 + ML

## 研究问题

设备在局部时间范围内的整体行为统计，是否比单包更有辨识度。

## 样本定义

• 一个样本 = 一个时间窗口

• 先按 device_id 分组

• 每个设备内部按 ts 排序

• 按固定时间窗口切分

## 模型

• Random Forest

• XGBoost

## 输入特征

### 窗口规模与长度分布

**pkt_count：**窗口内总包数，反映该时间段的活跃程度。

**total_bytes：**窗口内总字节数。

**total_payload_bytes：**窗口内总载荷字节数。

**packet_len_mean / std / min / max / median：**窗口内包长度分布统计。

**unique_packet_len_count：**窗口内不同包长的数量，刻画包长模式的多样性。

**payload_len_mean / std / min / max：**窗口内载荷长度统计。

### 时间与方向统计

**iat_mean / std / min / max：**窗口内包间隔分布统计。

**outbound_count / inbound_count：**出站与入站包数量。

**outbound_ratio：**出站包占比。

### 协议、控制位与对象多样性

**tcp_count / udp_count / icmp_count：**窗口内不同协议包数量。

**tcp_ratio / udp_ratio：**协议比例。

**unique_protocol_count：**协议类型多样性。

**syn_count / ack_count / fin_count / rst_count / psh_count：**TCP 控制行为统计。

**src_port_unique_count / dst_port_unique_count：**端口多样性。

**well_known_dst_port_ratio：**常见服务端口占比。

**unique_peer_ip_count：**通信对象多样性。

**https_port_ratio：**80/443 相关流量占比。

### 负载熵与 burst 行为

**payload_entropy_mean / std：**窗口内载荷熵统计。

**burst_count：**窗口内 burst 数量。

**burst_rate：**单位时间内 burst 数量。

**burst_packet_ratio / burst_byte_ratio：**属于 burst 的包或字节占比。

**mean/std/max_burst_packet_count：**每个 burst 包数分布。

**mean/std/max_burst_bytes：**每个 burst 字节数分布。

**mean/std/max_burst_duration：**每个 burst 持续时间分布。

**mean/std/max_inter_burst_gap：**burst 之间时间间隔分布。

# 六、实验组 C：序列压缩特征 + ML

## 研究问题

在不使用 RNN 的情况下，弱顺序结构是否已经足够帮助设备识别。

## 样本定义

• 一个样本 = 一个定长包序列

• 先按 device_id 分组

• 每个设备内部按 ts 排序

• 按固定包数切分

## 模型

• Random Forest

• XGBoost

## 输入特征

### 全局统计

**seq_packet_len_mean / std / min / max：**整个序列的包长统计。

**seq_payload_len_mean / std：**整个序列的载荷长度统计。

**seq_iat_mean / std：**整个序列的包间隔统计。

**seq_outbound_ratio：**整个序列的出站占比。

### 分段统计（前 1/3、中 1/3、后 1/3）

**packet_len_mean_seg1/2/3：**三个位置段的包长均值。

**payload_len_mean_seg1/2/3：**三个位置段的载荷均值。

**iat_mean_seg1/2/3：**三个位置段的时间间隔均值。

**outbound_ratio_seg1/2/3：**三个位置段的方向比例。

### 转移与首尾差异

**direction_switch_count：**方向切换次数。

**proto_switch_count：**协议切换次数。

**dst_port_change_count：**目标端口变化次数。

**tcp_flag_change_count：**TCP 控制状态变化次数。

**first_last_packet_len_diff：**首尾包长差异。

**first_last_payload_len_diff：**首尾载荷长度差异。

**first_last_direction_diff：**首尾方向差异。

# 七、实验组 D：原始序列级 + RNN

## 研究问题

显式逐包顺序建模，是否比统计聚合和弱顺序摘要更好。

## 样本定义

• 一个样本 = 一个定长包序列

• 与实验组 C 使用相同的序列构造规则

## 模型

• LSTM

• GRU

## 输入特征

### 每个时间步的输入特征

**packet_len / payload_len / has_payload / payload_entropy：**描述逐包的大小与负载。

**delta_t：**描述逐包的局部节奏。

**ip_proto / src_port / dst_port / ip_ttl / eth_type：**描述逐包的协议、端口与栈行为。

**ip_len / ip_flags / ip_tos：**补充 IP 层结构信息。

**direction：**描述方向序列。

**tcp_syn / tcp_ack / tcp_fin / tcp_rst / tcp_psh / tcp_urg / tcp_window / tcp_dataofs：**描述 TCP 控制与栈行为序列。

**udp_len / icmp_type / icmp_code：**补充 UDP / ICMP 行为序列。

# 八、类别平衡策略

设置下限，剔除样本过于少的设备数据

验证集和测试集保持原始分布，不做强制平衡。

# 九、参数设计

• 窗口大小 W ，步长 S，序列大小H，序列步长L

## 窗口参数设计

• 阶段 A：固定s，比较不同 W 比1 min、5 min、15 min，用于分析不同时间尺度下的识别能力。

• 阶段 B：固定最佳 W\*，比较 不同S，用于分析重叠程度对效果、样本量和稳定性的影响。

## 序列参数设计

• 序列按固定包数切分，更适合 RNN，也更适合做最小上下文分析。

• 阶段 A：固定 L ，比较不同H比如10包、20包、50包，用于分析需要多长顺序上下文才能稳定识别设备。

• 阶段 B：固定最佳 L，比较 不同H，用于分析序列样本重叠程度的影响。

# 十、评价指标与时间成本

## 分类性能指标

• Accuracy

• Macro-F1

• Weighted-F1

• Per-class precision

• Per-class recall

• Confusion matrix

## 时间成本指标

• Feature construction time

• Training time

• Inference time total

• Inference time per sample

# 十一、特征重要性分析

不要在项目开始时就按重要性删特征。

主实验完成后，对三组 ML 实验分别做特征重要性分析：单包级、窗口级、序列压缩级。

作用包括：解释不同粒度下哪些特征最关键、分析冗余、并在必要时作为轻量补充筛选实验的依据。

# 十二、需要输出的数据与结果

## 数据构造阶段输出

• packet-level CSV：每个数据集一份，一行代表一个包。

• window-level 样本表：每个窗口一行，含标签、窗口特征及窗口参数 W、S。

• sequence-compressed 样本表：每个序列一行，含标签、序列压缩特征及参数 L、H。

• sequence-level 张量或索引文件：保存序列 ID、标签、逐包特征矩阵及参数 L、H。

## 模型实验输出

• 总体性能表：Representation、Model、Accuracy、Macro-F1、Weighted-F1。

• 每类性能表：Device class、Precision、Recall、F1。

• 混淆矩阵：保存数值矩阵与热力图图像。

• 时间成本表：Feature construction time、Training time、Inference time total。

## 参数敏感性实验输出

• 窗口参数实验：不同 W、不同 S 对应的 Accuracy、Macro-F1、Weighted-F1 和时间成本。

• 序列参数实验：不同 L、不同 H 对应的 Accuracy、Macro-F1、Weighted-F1 和时间成本。

## 特征重要性分析输出

• 特征重要性排序表。。

• 重要性柱状图。

# 十三、推荐执行顺序

• 阶段 1：生成 packet-level CSV，建立 device_id，划分 train / validation / test。

• 阶段 2：完成四组主实验：Packet + RF/XGB、Window + RF/XGB、Sequence-compressed + RF/XGB、Sequence + LSTM/GRU。

• 阶段 3：完成窗口参数与序列参数敏感性分析。

• 阶段 4：完成单包级、窗口级、序列压缩级的特征重要性分析。

• 阶段 5：汇总总体性能表、每类性能表、混淆矩阵、时间成本表、参数敏感性图和重要性图。