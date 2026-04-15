"""
基于 UNSW-IoTraffic 设备级 PCAP 的跨数据集验证。

思路：
1. 从 data/pcaps 中选取与当前 18 类标签重叠的设备
2. 对每个设备 PCAP 仅截取前一段连续流量（按包顺序取前 N 个已识别包）
3. 生成 packet-level CSV（全部标记为 test）
4. 复用现有跨数据集评估逻辑运行代表性模型
"""
import argparse
import logging
import os
import sys
import socket
import time

import dpkt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import cross_dataset_eval as cde
from src.preprocessing.pcap_loader import _mac_bytes_to_str, _shannon_entropy, _ip_to_str
from src.utils.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


PCAP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pcaps")
SUBSET_CSV_PATH = os.path.join(PROCESSED_DIR, "unsw_iotraffic_subset_packet_level.csv")

# 与当前论文标签体系直接重叠的设备
PCAP_FILE_TO_LABEL = {
    "AmazonEcho_44650d56ccd3.pcap": ("44:65:0d:56:cc:d3", "Amazon Echo"),
    "BelkinWemoMotionSensor_ec1a59832811.pcap": ("ec:1a:59:83:28:11", "Belkin wemo motion sensor"),
    "BelkinWemoSwitch_ec1a5979f489.pcap": ("ec:1a:59:79:f4:89", "Belkin Wemo switch"),
    "HPPrinter_705a0fe49bc0.pcap": ("70:5a:0f:e4:9b:c0", "HP Printer"),
    "NetatmoWeatherStation_70ee5003b8ac.pcap": ("70:ee:50:03:b8:ac", "Netatmo weather station"),
    "NetatmoWelcome_70ee50183443.pcap": ("70:ee:50:18:34:43", "Netatmo Welcome"),
    "NestDropCam_308cfb2fe4b2.pcap": ("30:8c:fb:2f:e4:b2", "Dropcam"),
    "PixStarPhotoFrame_e076d033bb85.pcap": ("e0:76:d0:33:bb:85", "PIX-STAR Photo-frame"),
    "SamsungCamera_00166cab6b88.pcap": ("00:16:6c:ab:6b:88", "Samsung SmartCam"),
    "SamsungSmartThings_d052a800675e.pcap": ("d0:52:a8:00:67:5e", "Smart Things"),
    "TPLinkCamera_f4f26D9351f1.pcap": ("f4:f2:6d:93:51:f1", "TP-Link Day Night Cloud camera"),
    "TPLinkSmartPlug_50c7bf005639.pcap": ("50:c7:bf:00:56:39", "TP-Link Smart plug"),
    "TribySpeaker_18B79E022044.pcap": ("18:b7:9e:02:20:44", "Triby Speaker"),
    "WithingsBabyMonitor_0024e41118a8.pcapng": ("00:24:e4:11:18:a8", "Withings Smart Baby Monitor"),
}


def _iter_pcap(path):
    with open(path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
            for item in reader:
                yield item
        except ValueError:
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)
            for item in reader:
                yield item


def _parse_single_device_pcap(pcap_path, device_mac, device_name, max_packets):
    rows = []
    packet_id = 0

    for ts, buf in _iter_pcap(pcap_path):
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
        device_mac_lower = device_mac.lower()

        if src_mac != device_mac_lower and dst_mac != device_mac_lower:
            continue

        direction = 0 if src_mac == device_mac_lower else 1

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

        if dst_port < 1024:
            dport_class = 0
        elif dst_port < 49152:
            dport_class = 1
        else:
            dport_class = 2

        rows.append({
            "dataset_name": "unsw_iotraffic_subset",
            "capture_id": os.path.basename(pcap_path),
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
            "eth_type": eth.type,
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
            "is_https": int(dst_port in (80, 443) or src_port in (80, 443)),
            "is_well_known_port": int(dst_port < 1024),
            "split": "test",
        })
        packet_id += 1
        if packet_id >= max_packets:
            break

    return rows


def build_subset_csv(max_packets_per_device=50000):
    all_rows = []
    for filename, (mac, label) in PCAP_FILE_TO_LABEL.items():
        pcap_path = os.path.join(PCAP_DIR, filename)
        if not os.path.exists(pcap_path):
            logger.warning("缺失文件，跳过: %s", pcap_path)
            continue
        t0 = time.time()
        rows = _parse_single_device_pcap(pcap_path, mac, label, max_packets_per_device)
        logger.info("%s -> %d packets parsed in %.1fs", filename, len(rows), time.time() - t0)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("UNSW-IoTraffic 子集解析结果为空")

    df = pd.DataFrame(all_rows)
    df.sort_values(["device_id", "ts"], inplace=True)
    df["delta_t"] = df.groupby("device_id")["ts"].diff().fillna(0.0)
    df.to_csv(SUBSET_CSV_PATH, index=False)
    logger.info("UNSW-IoTraffic 子集 CSV 已保存: %s (%d rows, %d devices)",
                SUBSET_CSV_PATH, len(df), df["device_id"].nunique())
    return SUBSET_CSV_PATH


def run_subset_cross_eval(csv_path, groups=None):
    original_path = cde.NEW_CSV_PATH
    try:
        cde.NEW_CSV_PATH = csv_path
        selected = set(groups or ["A", "B", "C", "D", "E0", "E1"])
        results = []
        for group, runner, models in [
            ("A", cde.cross_dataset_group_a, ("rf", "xgb")),
            ("B", cde.cross_dataset_group_b, ("rf", "xgb")),
            ("C", cde.cross_dataset_group_c, ("rf", "xgb")),
            ("D", cde.cross_dataset_group_d, ("lstm", "gru")),
            ("E0", cde.cross_dataset_group_e, ("rf", "xgb")),
            ("E1", cde.cross_dataset_group_e1, ("rf", "xgb")),
        ]:
            if group not in selected:
                continue
            for model in models:
                logger.info("Running %s-%s", group, model)
                m = runner(model)
                if m:
                    results.append({
                        "group": group,
                        "model": model,
                        "accuracy": m["accuracy"],
                        "macro_f1": m["macro_f1"],
                        "weighted_f1": m["weighted_f1"],
                    })
        df = pd.DataFrame(results)
        out_path = os.path.join(PROCESSED_DIR, "unsw_iotraffic_subset_cross_summary.csv")
        df.to_csv(out_path, index=False)
        logger.info("subset cross summary saved: %s", out_path)
        return out_path, df
    finally:
        cde.NEW_CSV_PATH = original_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_packets_per_device", type=int, default=50000)
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--groups", type=str, default="A,B,C,D,E0,E1")
    args = parser.parse_args()

    csv_path = build_subset_csv(max_packets_per_device=args.max_packets_per_device)
    if args.build_only:
        return
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    out_path, df = run_subset_cross_eval(csv_path, groups=groups)
    print(out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
