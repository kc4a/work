"""
Group B: 窗口级特征计算
输入一个窗口的 DataFrame，输出一行特征 dict
"""
import numpy as np
import pandas as pd

from src.utils.config import BURST_GAP_THRESHOLD


def compute_window_features(win: pd.DataFrame) -> dict:
    """从一个时间窗口的包级 DataFrame 计算全部窗口级特征"""
    feat = {}
    n = len(win)

    # ── 窗口规模与长度分布 ────────────────────
    feat["pkt_count"] = n
    feat["total_bytes"] = win["packet_len"].sum()
    feat["total_payload_bytes"] = win["payload_len"].sum()

    plen = win["packet_len"]
    feat["packet_len_mean"] = plen.mean()
    feat["packet_len_std"] = plen.std(ddof=0) if n > 1 else 0.0
    feat["packet_len_min"] = plen.min()
    feat["packet_len_max"] = plen.max()
    feat["packet_len_median"] = plen.median()
    feat["unique_packet_len_count"] = plen.nunique()

    pllen = win["payload_len"]
    feat["payload_len_mean"] = pllen.mean()
    feat["payload_len_std"] = pllen.std(ddof=0) if n > 1 else 0.0
    feat["payload_len_min"] = pllen.min()
    feat["payload_len_max"] = pllen.max()

    # ── 时间与方向统计 ────────────────────────
    iat = win["delta_t"].iloc[1:] if n > 1 else pd.Series([0.0])
    # 重新计算窗口内 IAT（可能跨设备 delta_t 不准）
    if n > 1:
        ts_vals = win["ts"].values
        iat_vals = np.diff(ts_vals)
        iat_vals = np.maximum(iat_vals, 0.0)
    else:
        iat_vals = np.array([0.0])

    feat["iat_mean"] = float(np.mean(iat_vals))
    feat["iat_std"] = float(np.std(iat_vals)) if len(iat_vals) > 1 else 0.0
    feat["iat_min"] = float(np.min(iat_vals))
    feat["iat_max"] = float(np.max(iat_vals))

    dir_vals = win["direction"]
    feat["outbound_count"] = int((dir_vals == 0).sum())
    feat["inbound_count"] = int((dir_vals == 1).sum())
    feat["outbound_ratio"] = feat["outbound_count"] / n if n > 0 else 0.0

    # ── 协议、控制位与对象多样性 ──────────────
    proto = win["ip_proto"]
    feat["tcp_count"] = int((proto == 6).sum())
    feat["udp_count"] = int((proto == 17).sum())
    feat["icmp_count"] = int(((proto == 1) | (proto == 58)).sum())
    feat["tcp_ratio"] = feat["tcp_count"] / n if n > 0 else 0.0
    feat["udp_ratio"] = feat["udp_count"] / n if n > 0 else 0.0
    feat["unique_protocol_count"] = proto.nunique()

    feat["syn_count"] = int(win["tcp_syn"].sum())
    feat["ack_count"] = int(win["tcp_ack"].sum())
    feat["fin_count"] = int(win["tcp_fin"].sum())
    feat["rst_count"] = int(win["tcp_rst"].sum())
    feat["psh_count"] = int(win["tcp_psh"].sum())

    feat["src_port_unique_count"] = win["src_port"].nunique()
    feat["dst_port_unique_count"] = win["dst_port"].nunique()

    well_known = win["is_well_known_port"]
    feat["well_known_dst_port_ratio"] = float(well_known.mean())

    # unique_peer_ip_count: outbound 的 dst_ip + inbound 的 src_ip
    out_peers = win.loc[win["direction"] == 0, "dst_ip"].nunique() if "dst_ip" in win.columns else 0
    in_peers = win.loc[win["direction"] == 1, "src_ip"].nunique() if "src_ip" in win.columns else 0
    feat["unique_peer_ip_count"] = out_peers + in_peers

    https_col = win["is_https"]
    feat["https_port_ratio"] = float(https_col.mean())

    # ── 负载熵统计 ────────────────────────────
    ent = win["payload_entropy"]
    feat["payload_entropy_mean"] = ent.mean()
    feat["payload_entropy_std"] = ent.std(ddof=0) if n > 1 else 0.0

    # ── Burst 行为 ────────────────────────────
    _compute_burst_features(feat, iat_vals, win)

    return feat


def _compute_burst_features(feat: dict, iat_vals: np.ndarray, win: pd.DataFrame):
    """基于 IAT 阈值检测 burst 并计算相关统计"""
    n = len(win)
    threshold = BURST_GAP_THRESHOLD

    if len(iat_vals) == 0 or n <= 1:
        for k in [
            "burst_count", "burst_rate",
            "burst_packet_ratio", "burst_byte_ratio",
            "mean_burst_packet_count", "std_burst_packet_count", "max_burst_packet_count",
            "mean_burst_bytes", "std_burst_bytes", "max_burst_bytes",
            "mean_burst_duration", "std_burst_duration", "max_burst_duration",
            "mean_inter_burst_gap", "std_inter_burst_gap", "max_inter_burst_gap",
        ]:
            feat[k] = 0.0
        return

    # 检测 burst：连续 IAT < threshold 的包段
    is_in_burst = iat_vals < threshold  # len = n-1
    bursts = []
    pkt_lens = win["packet_len"].values
    ts_vals = win["ts"].values
    current_start = None

    for i in range(len(is_in_burst)):
        if is_in_burst[i]:
            if current_start is None:
                current_start = i  # burst 从第 i 个包开始 (实际是第 i 和 i+1)
        else:
            if current_start is not None:
                bursts.append((current_start, i + 1))  # [start, end) 包索引
                current_start = None
    if current_start is not None:
        bursts.append((current_start, len(is_in_burst)))

    burst_count = len(bursts)
    window_duration = ts_vals[-1] - ts_vals[0] if n > 1 else 1.0
    window_duration = max(window_duration, 1e-6)

    feat["burst_count"] = burst_count
    feat["burst_rate"] = burst_count / window_duration

    if burst_count == 0:
        for k in [
            "burst_packet_ratio", "burst_byte_ratio",
            "mean_burst_packet_count", "std_burst_packet_count", "max_burst_packet_count",
            "mean_burst_bytes", "std_burst_bytes", "max_burst_bytes",
            "mean_burst_duration", "std_burst_duration", "max_burst_duration",
            "mean_inter_burst_gap", "std_inter_burst_gap", "max_inter_burst_gap",
        ]:
            feat[k] = 0.0
        return

    burst_pkt_counts = []
    burst_byte_sums = []
    burst_durations = []

    for start, end in bursts:
        pkt_count = end - start + 1  # +1 因为 burst 包含 start 和 end 位置的包
        pkt_count = min(pkt_count, n)
        burst_pkt_counts.append(pkt_count)
        burst_byte_sums.append(float(pkt_lens[start:end + 1].sum()))
        burst_durations.append(ts_vals[min(end, n - 1)] - ts_vals[start])

    total_burst_pkts = sum(burst_pkt_counts)
    total_burst_bytes = sum(burst_byte_sums)

    feat["burst_packet_ratio"] = total_burst_pkts / n
    feat["burst_byte_ratio"] = total_burst_bytes / max(pkt_lens.sum(), 1)

    bpc = np.array(burst_pkt_counts, dtype=np.float64)
    feat["mean_burst_packet_count"] = float(bpc.mean())
    feat["std_burst_packet_count"] = float(bpc.std()) if len(bpc) > 1 else 0.0
    feat["max_burst_packet_count"] = float(bpc.max())

    bbs = np.array(burst_byte_sums, dtype=np.float64)
    feat["mean_burst_bytes"] = float(bbs.mean())
    feat["std_burst_bytes"] = float(bbs.std()) if len(bbs) > 1 else 0.0
    feat["max_burst_bytes"] = float(bbs.max())

    bd = np.array(burst_durations, dtype=np.float64)
    feat["mean_burst_duration"] = float(bd.mean())
    feat["std_burst_duration"] = float(bd.std()) if len(bd) > 1 else 0.0
    feat["max_burst_duration"] = float(bd.max())

    if burst_count > 1:
        burst_ends = [ts_vals[min(e, n - 1)] for _, e in bursts]
        burst_starts = [ts_vals[s] for s, _ in bursts]
        gaps = [burst_starts[i + 1] - burst_ends[i] for i in range(len(bursts) - 1)]
        gaps = np.array(gaps, dtype=np.float64)
        feat["mean_inter_burst_gap"] = float(gaps.mean())
        feat["std_inter_burst_gap"] = float(gaps.std()) if len(gaps) > 1 else 0.0
        feat["max_inter_burst_gap"] = float(gaps.max())
    else:
        feat["mean_inter_burst_gap"] = 0.0
        feat["std_inter_burst_gap"] = 0.0
        feat["max_inter_burst_gap"] = 0.0
