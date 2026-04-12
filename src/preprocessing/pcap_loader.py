"""
PCAP → packet-level CSV
一行 = 一个包，提取实验计划中定义的全部字段。
"""
import os
import math
import struct
import socket
import time
import logging

import dpkt
import numpy as np
import pandas as pd

from src.utils.config import (
    PCAP_PATH, PACKET_CSV_PATH, PROCESSED_DIR,
    MAC_TO_DEVICE, LOCAL_PREFIXES,
    MIN_SAMPLES_PER_DEVICE, TRAIN_RATIO, VAL_RATIO,
)

logger = logging.getLogger(__name__)


# ── 辅助函数 ──────────────────────────────────────

def _mac_bytes_to_str(mac_bytes: bytes) -> str:
    return ":".join(f"{b:02x}" for b in mac_bytes)


def _is_local_ip(ip_str: str) -> bool:
    return any(ip_str.startswith(p) for p in LOCAL_PREFIXES)


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.zeros(256, dtype=np.int64)
    for b in data:
        counts[b] += 1
    probs = counts[counts > 0] / len(data)
    return -float(np.sum(probs * np.log2(probs)))


def _ip_to_str(ip_pkt):
    if isinstance(ip_pkt, dpkt.ip.IP):
        return socket.inet_ntoa(ip_pkt.src), socket.inet_ntoa(ip_pkt.dst)
    elif isinstance(ip_pkt, dpkt.ip6.IP6):
        return (
            socket.inet_ntop(socket.AF_INET6, ip_pkt.src),
            socket.inet_ntop(socket.AF_INET6, ip_pkt.dst),
        )
    return None, None


# ── 核心解析 ──────────────────────────────────────

def parse_pcap_to_packet_csv(
    pcap_path: str = PCAP_PATH,
    output_path: str = PACKET_CSV_PATH,
    force: bool = False,
    mac_to_device: dict = None,
    min_samples: int = None,
    test_only: bool = False,
) -> pd.DataFrame:
    """解析 PCAP 文件，输出 packet-level CSV。如已缓存则直接加载。

    Parameters
    ----------
    mac_to_device : dict, optional
        自定义 MAC→设备名映射（默认使用 config 中的 MAC_TO_DEVICE）
    min_samples : int, optional
        最少包数过滤阈值（默认使用 MIN_SAMPLES_PER_DEVICE）；设为 0 跳过过滤
    test_only : bool
        若为 True，所有行 split="test"（用于跨数据集评估）
    """
    if mac_to_device is None:
        mac_to_device = MAC_TO_DEVICE
    if min_samples is None:
        min_samples = MIN_SAMPLES_PER_DEVICE

    if os.path.exists(output_path) and not force:
        logger.info("加载已缓存的 packet-level CSV: %s", output_path)
        return pd.read_csv(output_path)

    logger.info("开始解析 PCAP: %s", pcap_path)
    t0 = time.time()

    # dataset_name / capture_id 取自文件名
    basename = os.path.splitext(os.path.basename(pcap_path))[0]
    dataset_name = basename
    capture_id = basename

    rows = []
    packet_id = 0

    with open(pcap_path, "rb") as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            pcap = dpkt.pcapng.Reader(f)

        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.NeedData, dpkt.UnpackError):
                continue

            if isinstance(eth.data, dpkt.ip.IP):
                ip_pkt = eth.data
                ip_version = 4
            elif isinstance(eth.data, dpkt.ip6.IP6):
                ip_pkt = eth.data
                ip_version = 6
            else:
                continue

            src_ip_str, dst_ip_str = _ip_to_str(ip_pkt)
            if src_ip_str is None:
                continue

            src_mac = _mac_bytes_to_str(eth.src)
            dst_mac = _mac_bytes_to_str(eth.dst)

            device_name = mac_to_device.get(src_mac) or mac_to_device.get(dst_mac)
            if device_name is None:
                continue

            if src_mac in mac_to_device:
                direction = 0  # outbound
            else:
                direction = 1  # inbound

            # 传输层
            transport = ip_pkt.data
            src_port = dst_port = 0
            tcp_syn = tcp_ack = tcp_fin = tcp_rst = tcp_psh = tcp_urg = 0
            tcp_window = tcp_dataofs = 0
            udp_len_val = 0
            icmp_type_val = icmp_code_val = 0
            payload = b""

            proto = ip_pkt.p if ip_version == 4 else ip_pkt.nxt

            if isinstance(transport, dpkt.tcp.TCP):
                src_port = transport.sport
                dst_port = transport.dport
                flags = transport.flags
                tcp_syn = int(bool(flags & dpkt.tcp.TH_SYN))
                tcp_ack = int(bool(flags & dpkt.tcp.TH_ACK))
                tcp_fin = int(bool(flags & dpkt.tcp.TH_FIN))
                tcp_rst = int(bool(flags & dpkt.tcp.TH_RST))
                tcp_psh = int(bool(flags & dpkt.tcp.TH_PUSH))
                tcp_urg = int(bool(flags & dpkt.tcp.TH_URG))
                tcp_window = transport.win
                tcp_dataofs = transport.off
                payload = transport.data if isinstance(transport.data, bytes) else b""
            elif isinstance(transport, dpkt.udp.UDP):
                src_port = transport.sport
                dst_port = transport.dport
                udp_len_val = transport.ulen
                payload = transport.data if isinstance(transport.data, bytes) else b""
            elif isinstance(transport, (dpkt.icmp.ICMP, dpkt.icmp6.ICMP6)):
                icmp_type_val = transport.type
                icmp_code_val = transport.code
                payload = transport.data if isinstance(transport.data, bytes) else b""

            packet_len = len(buf)
            payload_len = len(payload)
            has_payload = int(payload_len > 0)
            payload_entropy = _shannon_entropy(payload) if has_payload else 0.0

            if ip_version == 4:
                ip_len = ip_pkt.len
                ip_ttl = ip_pkt.ttl
                ip_tos = ip_pkt.tos
                ip_flags = (ip_pkt.off >> 13) & 0x7
                ip_ihl = ip_pkt.hl
                ip_df = int(bool(ip_pkt.off & dpkt.ip.IP_DF))
            else:
                ip_len = ip_pkt.plen + 40
                ip_ttl = ip_pkt.hlim
                ip_tos = (ip_pkt.fc >> 20) & 0xFF
                ip_flags = 0
                ip_ihl = 0
                ip_df = 0

            eth_type = eth.type

            if dst_port < 1024:
                dport_class = 0
            elif dst_port < 49152:
                dport_class = 1
            else:
                dport_class = 2
            is_https = int(dst_port in (80, 443) or src_port in (80, 443))
            is_well_known_port = int(dst_port < 1024)

            rows.append({
                "dataset_name": dataset_name,
                "capture_id": capture_id,
                "packet_id": packet_id,
                "ts": ts,
                "device_id": device_name,
                "src_ip": src_ip_str,
                "dst_ip": dst_ip_str,
                "src_mac": src_mac,
                "dst_mac": dst_mac,
                "src_port": src_port,
                "dst_port": dst_port,
                "direction": direction,
                "packet_len": packet_len,
                "payload_len": payload_len,
                "has_payload": has_payload,
                "payload_entropy": payload_entropy,
                "eth_type": eth_type,
                "ip_version": ip_version,
                "ip_proto": proto,
                "ip_len": ip_len,
                "ip_ttl": ip_ttl,
                "ip_tos": ip_tos,
                "ip_flags": ip_flags,
                "ip_ihl": ip_ihl,
                "ip_df": ip_df,
                "tcp_syn": tcp_syn,
                "tcp_ack": tcp_ack,
                "tcp_fin": tcp_fin,
                "tcp_rst": tcp_rst,
                "tcp_psh": tcp_psh,
                "tcp_urg": tcp_urg,
                "tcp_window": tcp_window,
                "tcp_dataofs": tcp_dataofs,
                "udp_len": udp_len_val,
                "icmp_type": icmp_type_val,
                "icmp_code": icmp_code_val,
                "dport_class": dport_class,
                "is_https": is_https,
                "is_well_known_port": is_well_known_port,
            })
            packet_id += 1

            if packet_id % 500000 == 0:
                logger.info("  已解析 %d 个包 …", packet_id)

    logger.info("PCAP 解析完毕：共 %d 个已知设备包，耗时 %.1f s",
                len(rows), time.time() - t0)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("解析结果为空，请检查 PCAP 和 MAC 映射")

    # delta_t（按设备分组，按时间排序）
    df.sort_values(["device_id", "ts"], inplace=True)
    df["delta_t"] = df.groupby("device_id")["ts"].diff().fillna(0.0)

    # 过滤样本过少的设备
    if min_samples > 0:
        counts = df["device_id"].value_counts()
        keep_devices = counts[counts >= min_samples].index
        n_before = df["device_id"].nunique()
        df = df[df["device_id"].isin(keep_devices)].copy()
        logger.info("设备过滤: %d → %d (min_samples=%d)",
                    n_before, df["device_id"].nunique(), min_samples)

    # 时间序划分 train / val / test
    if test_only:
        df["split"] = "test"
        logger.info("跨数据集模式: 所有 %d 行标记为 test", len(df))
    else:
        def _assign_split(group):
            n = len(group)
            n_train = int(n * TRAIN_RATIO)
            n_val = int(n * (TRAIN_RATIO + VAL_RATIO))
            splits = (["train"] * n_train
                      + ["val"] * (n_val - n_train)
                      + ["test"] * (n - n_val))
            group = group.copy()
            group["split"] = splits
            return group

        split_parts = []
        for _, group in df.groupby("device_id", sort=False):
            split_parts.append(_assign_split(group))
        df = pd.concat(split_parts, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("packet-level CSV 已保存: %s  (%d 行)", output_path, len(df))

    for dev, cnt in df["device_id"].value_counts().items():
        split_dist = df[df["device_id"] == dev]["split"].value_counts().to_dict()
        logger.info("  %-40s  %6d  %s", dev, cnt, split_dist)

    return df


def load_packet_csv(csv_path: str = PACKET_CSV_PATH) -> pd.DataFrame:
    """加载 packet-level CSV"""
    if not os.path.exists(csv_path):
        logger.info("CSV 不存在，触发 PCAP 解析 …")
        return parse_pcap_to_packet_csv()
    return pd.read_csv(csv_path)
