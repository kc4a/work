"""
IoT 设备识别 — 全局配置
多粒度行为表示比较 + 早期识别 + 性能-成本权衡
"""
import os

# ── 路径 ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PCAP_PATH = os.path.join(BASE_DIR, "data", "16-09-23.pcap")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PACKET_CSV_PATH = os.path.join(PROCESSED_DIR, "packet_level.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for _d in [
    PROCESSED_DIR,
    os.path.join(RESULTS_DIR, "experiments"),
    os.path.join(RESULTS_DIR, "figures"),
    os.path.join(RESULTS_DIR, "tables"),
    os.path.join(RESULTS_DIR, "models"),
    os.path.join(RESULTS_DIR, "logs"),
]:
    os.makedirs(_d, exist_ok=True)

# ── 数据 / 划分 ──────────────────────────────────
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_SAMPLES_PER_DEVICE = 1000   # 低于此数量的设备被剔除

# 本地 IP 前缀（用于判断方向）
LOCAL_PREFIXES = ("192.168.", "172.17.", "fe80:", "2405:")

# MAC → 设备名映射（来自 device_list.md）
MAC_TO_DEVICE = {
    "d0:52:a8:00:67:5e": "Smart Things",
    "44:65:0d:56:cc:d3": "Amazon Echo",
    "70:ee:50:18:34:43": "Netatmo Welcome",
    "f4:f2:6d:93:51:f1": "TP-Link Day Night Cloud camera",
    "00:16:6c:ab:6b:88": "Samsung SmartCam",
    "30:8c:fb:2f:e4:b2": "Dropcam",
    "00:62:6e:51:27:2e": "Insteon Camera",
    "e8:ab:fa:19:de:4f": "Unknown Device",
    "00:24:e4:11:18:a8": "Withings Smart Baby Monitor",
    "ec:1a:59:79:f4:89": "Belkin Wemo switch",
    "50:c7:bf:00:56:39": "TP-Link Smart plug",
    "74:c6:3b:29:d7:1d": "iHome",
    "ec:1a:59:83:28:11": "Belkin wemo motion sensor",
    "18:b4:30:25:be:e4": "NEST Protect smoke alarm",
    "70:ee:50:03:b8:ac": "Netatmo weather station",
    "00:24:e4:1b:6f:96": "Withings Smart scale",
    "74:6a:89:00:2e:25": "Blipcare Blood Pressure meter",
    "00:24:e4:20:28:c6": "Withings Aura smart sleep sensor",
    "d0:73:d5:01:83:08": "Light Bulbs LiFX Smart Bulb",
    "18:b7:9e:02:20:44": "Triby Speaker",
    "e0:76:d0:33:bb:85": "PIX-STAR Photo-frame",
    "70:5a:0f:e4:9b:c0": "HP Printer",
    "08:21:ef:3b:fc:e3": "Samsung Galaxy Tab",
    "30:8c:fb:b6:ea:45": "Nest Dropcam",
    "40:f3:08:ff:1e:da": "Android Phone",
    "74:2f:68:81:69:42": "Laptop",
    "ac:bc:32:d4:6f:2f": "MacBook",
    "b4:ce:f6:a7:a3:c2": "Android Phone 2",
    "d0:a6:37:df:a1:e1": "IPhone",
    "f4:5c:89:93:cc:85": "MacBook/Iphone",
    "14:cc:20:51:33:ea": "TPLink Router Bridge LAN",
}

# ── Group A: 单包级特征 ──────────────────────────
GROUP_A_FEATURES = [
    "packet_len", "payload_len", "has_payload", "payload_entropy",
    "delta_t",
    "ip_proto", "src_port", "dst_port", "eth_type", "ip_version",
    "ip_len", "ip_ttl", "ip_tos", "ip_flags", "ip_ihl", "ip_df",
    "tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst", "tcp_psh", "tcp_urg",
    "tcp_window", "tcp_dataofs",
    "udp_len", "icmp_type", "icmp_code",
    "direction",
]

# ── Group B: 窗口级参数 ──────────────────────────
WINDOW_SIZES = [60, 300, 900]           # 秒 (1 min, 5 min, 15 min)
WINDOW_STEPS = [30, 60, 150, 300, 450]  # 步长候选
DEFAULT_WINDOW_SIZE = 300
DEFAULT_WINDOW_STEP = 300
MIN_PACKETS_PER_WINDOW = 5
BURST_GAP_THRESHOLD = 1.0              # 秒

# ── Group C/D: 序列级参数 ────────────────────────
SEQ_LENGTHS = [10, 20, 50]             # 包数
SEQ_STEPS = [5, 10, 20, 25, 50]       # 步长候选
DEFAULT_SEQ_LENGTH = 20
DEFAULT_SEQ_STEP = 20

# ── Group D: 每时间步特征 ────────────────────────
GROUP_D_FEATURES = [
    "packet_len", "payload_len", "has_payload", "payload_entropy",
    "delta_t",
    "ip_proto", "src_port", "dst_port", "ip_ttl", "eth_type",
    "ip_len", "ip_flags", "ip_tos",
    "direction",
    "tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst", "tcp_psh", "tcp_urg",
    "tcp_window", "tcp_dataofs",
    "udp_len", "icmp_type", "icmp_code",
]

# ── 模型超参数 ───────────────────────────────────
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3
RNN_BATCH_SIZE = 32
RNN_LEARNING_RATE = 1e-3
RNN_EPOCHS = 50
RNN_PATIENCE = 10

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu"

# ── 评估 ─────────────────────────────────────────
INFERENCE_REPEAT = 100
