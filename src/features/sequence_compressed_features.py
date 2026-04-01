"""
Group C: 序列压缩特征计算
输入一个定长包序列 DataFrame，输出一行特征 dict
"""
import numpy as np
import pandas as pd


def compute_sequence_compressed_features(seq: pd.DataFrame) -> dict:
    """从定长包序列计算全局统计 + 分段统计 + 转移特征"""
    feat = {}
    n = len(seq)

    # ── 全局统计 ──────────────────────────────
    plen = seq["packet_len"].values.astype(np.float64)
    feat["seq_packet_len_mean"] = float(np.mean(plen))
    feat["seq_packet_len_std"] = float(np.std(plen)) if n > 1 else 0.0
    feat["seq_packet_len_min"] = float(np.min(plen))
    feat["seq_packet_len_max"] = float(np.max(plen))

    pllen = seq["payload_len"].values.astype(np.float64)
    feat["seq_payload_len_mean"] = float(np.mean(pllen))
    feat["seq_payload_len_std"] = float(np.std(pllen)) if n > 1 else 0.0

    ts_arr = seq["ts"].values
    if n > 1:
        iat = np.diff(ts_arr).astype(np.float64)
        iat = np.maximum(iat, 0.0)
    else:
        iat = np.array([0.0])
    feat["seq_iat_mean"] = float(np.mean(iat))
    feat["seq_iat_std"] = float(np.std(iat)) if len(iat) > 1 else 0.0

    direction = seq["direction"].values
    feat["seq_outbound_ratio"] = float(np.mean(direction == 0))

    # ── 分段统计（前 1/3、中 1/3、后 1/3）────
    seg_size = max(1, n // 3)
    segments = [
        seq.iloc[:seg_size],
        seq.iloc[seg_size:2 * seg_size],
        seq.iloc[2 * seg_size:],
    ]

    for i, seg in enumerate(segments, 1):
        suffix = f"_seg{i}"
        if len(seg) == 0:
            feat[f"packet_len_mean{suffix}"] = 0.0
            feat[f"payload_len_mean{suffix}"] = 0.0
            feat[f"iat_mean{suffix}"] = 0.0
            feat[f"outbound_ratio{suffix}"] = 0.0
            continue

        feat[f"packet_len_mean{suffix}"] = float(seg["packet_len"].mean())
        feat[f"payload_len_mean{suffix}"] = float(seg["payload_len"].mean())

        seg_ts = seg["ts"].values
        if len(seg_ts) > 1:
            seg_iat = np.diff(seg_ts).astype(np.float64)
            feat[f"iat_mean{suffix}"] = float(np.mean(np.maximum(seg_iat, 0.0)))
        else:
            feat[f"iat_mean{suffix}"] = 0.0

        feat[f"outbound_ratio{suffix}"] = float(np.mean(seg["direction"].values == 0))

    # ── 转移与首尾差异 ───────────────────────
    feat["direction_switch_count"] = int(np.sum(np.diff(direction) != 0))

    proto = seq["ip_proto"].values
    feat["proto_switch_count"] = int(np.sum(np.diff(proto) != 0))

    dport = seq["dst_port"].values
    feat["dst_port_change_count"] = int(np.sum(np.diff(dport) != 0))

    # TCP flag 变化次数：任一 flag 改变即算一次
    tcp_flags = seq[["tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst", "tcp_psh", "tcp_urg"]].values
    if n > 1:
        flag_diff = np.diff(tcp_flags, axis=0)
        feat["tcp_flag_change_count"] = int(np.sum(np.any(flag_diff != 0, axis=1)))
    else:
        feat["tcp_flag_change_count"] = 0

    feat["first_last_packet_len_diff"] = float(plen[-1] - plen[0])
    feat["first_last_payload_len_diff"] = float(pllen[-1] - pllen[0])
    feat["first_last_direction_diff"] = int(direction[-1] != direction[0])

    return feat
