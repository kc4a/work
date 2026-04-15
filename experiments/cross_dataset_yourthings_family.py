"""
YourThings 家族级跨数据集验证。
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import socket
import sys
import time
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.clf_metrics import compute_metrics
from src.features.sequence_compressed_features import compute_sequence_compressed_features
from src.features.window_features import compute_window_features
from src.models.classifier import IoTClassifier
from src.models.rnn_classifier import IoTRNNClassifier, RNNTrainer, create_dataloaders
from src.preprocessing.group_e1_loader import (
    CUR_PACKET_FIELDS,
    HIST_BASELINE_KEYS,
    GroupE1Loader,
    _compute_relation_features,
)
from src.preprocessing.pcap_loader import _ip_to_str, _mac_bytes_to_str, _shannon_entropy, load_packet_csv
from src.utils.config import (
    BASE_DIR,
    DEVICE,
    DEFAULT_GROUP_E1_D_MAX,
    DEFAULT_GROUP_E_HIST_WINDOW,
    DEFAULT_GROUP_E_SAMPLE_STEP,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STEP,
    GROUP_A_FEATURES,
    GROUP_D_FEATURES,
    PROCESSED_DIR,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

YOURTHINGS_DIR = Path(BASE_DIR) / "data" / "2018" / "03" / "20"
DEVICE_MAPPING_PATH = Path(BASE_DIR) / "device_mapping.csv"
YOURTHINGS_FAMILY_CSV = Path(PROCESSED_DIR) / "yourthings_family_packet_level.csv"
TABLE_DIR = Path(RESULTS_DIR) / "tables"
EXPR_DIR = Path(RESULTS_DIR) / "experiments"
MODEL_DIR = Path(RESULTS_DIR) / "models"

TRAIN_DEVICE_TO_FAMILY = {
    "Amazon Echo": "speaker",
    "Triby Speaker": "speaker",
    "iHome": "speaker",
    "Smart Things": "hub",
    "Belkin Wemo switch": "plug_switch",
    "TP-Link Smart plug": "plug_switch",
    "Belkin wemo motion sensor": "motion_sensor",
    "Withings Smart Baby Monitor": "baby_monitor",
    "Dropcam": "camera",
    "Nest Dropcam": "camera",
    "Samsung SmartCam": "camera",
    "TP-Link Day Night Cloud camera": "camera",
    "Netatmo Welcome": "camera",
    "Insteon Camera": "camera",
    "Light Bulbs LiFX Smart Bulb": "light",
}

YOURTHINGS_DEVICE_TO_FAMILY = {
    "SamsungSmartThingsHub": "hub",
    "PhilipsHUEHub": "hub",
    "InsteonHub": "hub",
    "SecurifiAlmond": "hub",
    "WinkHub": "hub",
    "MiCasaVerdeVeraLite": "hub",
    "LogitechHarmonyHub": "hub",
    "CasetaWirelessHub": "hub",
    "Wink2Hub": "hub",
    "Sonos": "speaker",
    "AmazonEchoGen1": "speaker",
    "GoogleHomeMini": "speaker",
    "GoogleHome": "speaker",
    "BoseSoundTouch10": "speaker",
    "HarmonKardonInvoke": "speaker",
    "AppleHomePod": "speaker",
    "AmazonEchoDotGen3": "speaker",
    "SonosBeam": "speaker",
    "BelkinWeMoSwitch": "plug_switch",
    "TP-LinkWiFiPlug": "plug_switch",
    "BelkinWeMoLink": "plug_switch",
    "BelkinWeMoMotionSensor": "motion_sensor",
    "WithingsHome": "baby_monitor",
    "NestCamera": "camera",
    "BelkinNetcam": "camera",
    "RingDoorbell": "camera",
    "NetgearArloCamera": "camera",
    "D-LinkDCS-5009LCamera": "camera",
    "LogitechLogiCircle": "camera",
    "Canary": "camera",
    "PiperNV": "camera",
    "ChineseWebcam": "camera",
    "AugustDoorbellCam": "camera",
    "NestCamIQ": "camera",
    "FacebookPortal": "camera",
    "AxisNetworkCamera": "camera",
    "LIFXVirtualBulb": "light",
    "KoogeekLightbulb": "light",
    "TP-LinkSmartWiFiLEDBulb": "light",
}

GROUP_C_CONFIG = {
    "rf": {"seq_length": 50, "step": 10},
    "xgb": {"seq_length": 50, "step": 25},
}

GROUP_D_CONFIG = {
    "lstm": {"seq_length": 50, "step": 10},
    "gru": {"seq_length": 50, "step": 10},
}


def _build_classifier(model_type: str) -> IoTClassifier:
    return IoTClassifier(model_type, n_jobs=1)


def _iter_pcap(path: Path):
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


def _load_yourthings_ip_maps():
    ip_to_device = {}
    ip_to_family = {}
    with open(DEVICE_MAPPING_PATH, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) != 2:
                continue
            device_name, ip_addr = row[0].strip(), row[1].strip()
            family = YOURTHINGS_DEVICE_TO_FAMILY.get(device_name)
            if family:
                ip_to_device[ip_addr] = device_name
                ip_to_family[ip_addr] = family
    return ip_to_device, ip_to_family


def build_yourthings_family_csv(
    max_packets_per_family: int = 50000,
    min_packets_per_family: int = 1000,
    max_files: int | None = None,
    force: bool = False,
) -> Path:
    if YOURTHINGS_FAMILY_CSV.exists() and not force:
        return YOURTHINGS_FAMILY_CSV

    ip_to_device, ip_to_family = _load_yourthings_ip_maps()
    files = sorted(p for p in YOURTHINGS_DIR.iterdir() if p.is_file())
    if max_files:
        files = files[:max_files]

    rows = []
    family_counts = {}
    packet_id = 0

    for file_idx, pcap_path in enumerate(files, 1):
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

            device_name = ip_to_device.get(src_ip_str)
            family = ip_to_family.get(src_ip_str)
            direction = 0
            if family is None:
                device_name = ip_to_device.get(dst_ip_str)
                family = ip_to_family.get(dst_ip_str)
                direction = 1
            if family is None:
                continue
            if family_counts.get(family, 0) >= max_packets_per_family:
                continue

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
                "dataset_name": "yourthings_family",
                "capture_id": pcap_path.name,
                "packet_id": packet_id,
                "ts": ts,
                "device_id": device_name,
                "family_id": family,
                "src_ip": src_ip_str,
                "dst_ip": dst_ip_str,
                "src_mac": _mac_bytes_to_str(eth.src),
                "dst_mac": _mac_bytes_to_str(eth.dst),
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
            family_counts[family] = family_counts.get(family, 0) + 1
            packet_id += 1

        logger.info(
            "Parsed %d/%d files, family counts: %s",
            file_idx, len(files),
            ", ".join(f"{k}={v}" for k, v in sorted(family_counts.items())),
        )

    if not rows:
        raise RuntimeError("YourThings family parsing produced no rows.")

    df = pd.DataFrame(rows)
    keep_families = [k for k, v in df["family_id"].value_counts().items() if v >= min_packets_per_family]
    df = df[df["family_id"].isin(keep_families)].copy()
    df.sort_values(["device_id", "ts"], inplace=True)
    df["delta_t"] = df.groupby("device_id")["ts"].diff().fillna(0.0)
    df.to_csv(YOURTHINGS_FAMILY_CSV, index=False)

    counts_df = (
        df.groupby("family_id")
        .agg(packets=("family_id", "size"), devices=("device_id", "nunique"))
        .reset_index()
        .sort_values("packets", ascending=False)
    )
    counts_df.to_csv(TABLE_DIR / "yourthings_family_test_counts.csv", index=False)
    logger.info("Saved %s (%d rows, %d families)", YOURTHINGS_FAMILY_CSV, len(df), df["family_id"].nunique())
    return YOURTHINGS_FAMILY_CSV


def _prepare_train_family_df() -> pd.DataFrame:
    old_df = load_packet_csv().copy()
    old_df["family_id"] = old_df["device_id"].map(TRAIN_DEVICE_TO_FAMILY)
    old_df = old_df[old_df["family_id"].notna()].copy()
    counts_df = (
        old_df.groupby(["split", "family_id"])
        .size()
        .reset_index(name="packets")
        .sort_values(["split", "packets"], ascending=[True, False])
    )
    counts_df.to_csv(TABLE_DIR / "yourthings_family_train_counts.csv", index=False)
    return old_df


def _load_test_family_df(csv_path: Path) -> pd.DataFrame:
    return load_packet_csv(str(csv_path)).copy()


def _common_families(old_df: pd.DataFrame, new_df: pd.DataFrame) -> list[str]:
    return sorted(set(old_df["family_id"].unique()) & set(new_df["family_id"].unique()))


def _build_windows_family(df: pd.DataFrame, split_filter: str | None, families: list[str], window_size: int, step: int):
    from src.utils.config import MIN_PACKETS_PER_WINDOW

    subset = df[df["family_id"].isin(families)].copy()
    if split_filter:
        subset = subset[subset["split"] == split_filter].copy()
    rows = []
    for device_id, dev_df in subset.groupby("device_id"):
        dev_df = dev_df.sort_values("ts").reset_index(drop=True)
        ts_arr = dev_df["ts"].values
        if len(ts_arr) == 0:
            continue
        family = dev_df["family_id"].iloc[0]
        t_start, t_end = ts_arr[0], ts_arr[-1]
        win_start = t_start
        while win_start < t_end:
            win_end = win_start + window_size
            mask = (ts_arr >= win_start) & (ts_arr < win_end)
            win_df = dev_df.loc[mask]
            if len(win_df) >= MIN_PACKETS_PER_WINDOW:
                feat = compute_window_features(win_df)
                feat["family_id"] = family
                rows.append(feat)
            win_start += step
    return pd.DataFrame(rows)


def _build_seq_compressed_family(df: pd.DataFrame, split_filter: str | None, families: list[str], seq_length: int, step: int):
    subset = df[df["family_id"].isin(families)].copy()
    if split_filter:
        subset = subset[subset["split"] == split_filter].copy()
    rows = []
    labels = []
    for device_id, dev_df in subset.groupby("device_id"):
        dev_df = dev_df.sort_values("ts").reset_index(drop=True)
        family = dev_df["family_id"].iloc[0]
        n = len(dev_df)
        start = 0
        while start + seq_length <= n:
            rows.append(compute_sequence_compressed_features(dev_df.iloc[start:start + seq_length]))
            labels.append(family)
            start += step
    return pd.DataFrame(rows).fillna(0.0), labels


def _build_raw_sequences_family(df: pd.DataFrame, split_filter: str | None, families: list[str], seq_length: int, step: int):
    subset = df[df["family_id"].isin(families)].copy()
    if split_filter:
        subset = subset[subset["split"] == split_filter].copy()
    arrays = []
    labels = []
    for device_id, dev_df in subset.groupby("device_id"):
        dev_df = dev_df.sort_values("ts").reset_index(drop=True)
        family = dev_df["family_id"].iloc[0]
        values = dev_df[GROUP_D_FEATURES].fillna(0.0).values.astype(np.float32)
        n = len(values)
        start = 0
        while start + seq_length <= n:
            arrays.append(values[start:start + seq_length])
            labels.append(family)
            start += step
    if not arrays:
        return np.empty((0, seq_length, len(GROUP_D_FEATURES)), dtype=np.float32), []
    return np.stack(arrays, axis=0), labels


def _build_group_e_family(df: pd.DataFrame, split_filter: str | None, families: list[str], sample_step: int, hist_window: int):
    from src.preprocessing.group_e_loader import _get_empty_window_features, _init_window_feature_keys

    seed_df = df.head(10).copy()
    if seed_df.empty:
        return pd.DataFrame()
    _init_window_feature_keys(seed_df)

    subset = df[df["family_id"].isin(families)].copy()
    if split_filter:
        subset = subset[subset["split"] == split_filter].copy()

    rows = []
    for device_id, dev_df in subset.groupby("device_id"):
        dev_df = dev_df.sort_values("ts").reset_index(drop=True)
        family = dev_df["family_id"].iloc[0]
        ts_arr = dev_df["ts"].values
        if len(ts_arr) < 2:
            continue
        t_start, t_end = ts_arr[0], ts_arr[-1]
        anchor = t_start + sample_step
        while anchor <= t_end:
            idx = np.searchsorted(ts_arr, anchor, side="left")
            if idx >= len(ts_arr):
                break
            t_cur = ts_arr[idx]
            cur_pkt = dev_df.iloc[idx]

            pkt_feat = {f"pkt_{f}": float(cur_pkt.get(f, 0.0)) for f in GROUP_A_FEATURES}
            hist_start = t_cur - hist_window
            hist_mask = (ts_arr >= hist_start) & (ts_arr < t_cur)
            hist_df = dev_df.loc[hist_mask]
            n_hist = len(hist_df)
            if n_hist >= 1:
                win_feat = compute_window_features(hist_df)
            else:
                win_feat = _get_empty_window_features()
            hist_feat = {f"hist_{k}": float(v) for k, v in win_feat.items()}

            if n_hist >= 2:
                actual_dur = float(ts_arr[hist_mask][-1] - ts_arr[hist_mask][0])
            else:
                actual_dur = 0.0
            total_bytes = float(hist_df["packet_len"].sum()) if n_hist > 0 else 0.0
            safe_dur = max(actual_dur, 1e-6)
            cold_feat = {
                "hist_duration_actual": actual_dur,
                "hist_window_ratio": actual_dur / hist_window,
                "hist_complete_flag": 1.0 if actual_dur >= hist_window * 0.9 else 0.0,
                "hist_packet_count_actual": float(n_hist),
                "packet_rate": n_hist / safe_dur,
                "byte_rate": total_bytes / safe_dur,
            }

            row = {}
            row.update(pkt_feat)
            row.update(hist_feat)
            row.update(cold_feat)
            row["family_id"] = family
            rows.append(row)
            anchor += sample_step
    return pd.DataFrame(rows)


def _build_group_e1_family(df: pd.DataFrame, split_filter: str | None, families: list[str], sample_step: int, hist_window: int, d_max: int):
    subset = df[df["family_id"].isin(families)].copy()
    if split_filter:
        subset = subset[subset["split"] == split_filter].copy()

    rows = []
    for device_id, dev_df in subset.groupby("device_id"):
        dev_df = dev_df.sort_values("ts").reset_index(drop=True)
        family = dev_df["family_id"].iloc[0]
        ts_arr = dev_df["ts"].values
        payload_arr = dev_df["payload_len"].values
        if len(ts_arr) < 2:
            continue
        t_start, t_end = ts_arr[0], ts_arr[-1]
        anchor = t_start + sample_step
        while anchor <= t_end:
            cur_idx = GroupE1Loader._select_packet(ts_arr, payload_arr, anchor, d_max)
            if cur_idx is None:
                anchor += sample_step
                continue
            t_cur = ts_arr[cur_idx]
            cur_pkt = dev_df.iloc[cur_idx]
            anchor_gap = anchor - t_cur

            cur_feat = {f"cur_{f}": float(cur_pkt.get(f, 0.0)) for f in CUR_PACKET_FIELDS}
            hist_start = t_cur - hist_window
            hist_mask = (ts_arr >= hist_start) & (ts_arr < t_cur)
            hist_df = dev_df.loc[hist_mask]
            n_hist = len(hist_df)
            if n_hist >= 1:
                win_stats = compute_window_features(hist_df)
            else:
                win_stats = {k: 0.0 for k in HIST_BASELINE_KEYS}
            hist_feat = {f"hist_{k}": float(win_stats.get(k, 0.0)) for k in HIST_BASELINE_KEYS}

            if n_hist >= 2:
                actual_dur = float(ts_arr[hist_mask][-1] - ts_arr[hist_mask][0])
            else:
                actual_dur = 0.0
            safe_dur = max(actual_dur, 1e-6)
            hist_feat["hist_packet_rate"] = n_hist / safe_dur
            hist_feat["hist_byte_rate"] = float(win_stats.get("total_bytes", 0.0)) / safe_dur

            rel_feat = _compute_relation_features(cur_pkt, win_stats, hist_df)
            ratio = actual_dur / hist_window if hist_window > 0 else 0.0
            if ratio < 0.3:
                stage = 0
            elif ratio < 0.8:
                stage = 1
            else:
                stage = 2
            reliability_feat = {
                "hist_duration_actual": actual_dur,
                "hist_window_ratio": ratio,
                "hist_packet_count_actual": float(n_hist),
                "hist_stage": float(stage),
                "anchor_gap": anchor_gap,
            }

            row = {}
            row.update(cur_feat)
            row.update(hist_feat)
            row.update(rel_feat)
            row.update(reliability_feat)
            row["family_id"] = family
            rows.append(row)
            anchor += sample_step
    return pd.DataFrame(rows)


def _save_metrics(tag: str, metrics: dict, group: str, model: str):
    summary = {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")}
    logger.info("%s => %s", tag, summary)
    return {"group": group, "model": model, **summary}


def run_group_a(old_df, new_df, families, model_type):
    tag = f"yourthings_family_group_a_{model_type}"
    train_df = old_df[(old_df["split"] == "train") & (old_df["family_id"].isin(families))].copy()
    test_df = new_df[new_df["family_id"].isin(families)].copy()
    le = LabelEncoder().fit(families)
    clf = _build_classifier(model_type)
    clf.train(train_df[GROUP_A_FEATURES].fillna(0.0).values.astype(np.float32), le.transform(train_df["family_id"]))
    y_pred = clf.predict(test_df[GROUP_A_FEATURES].fillna(0.0).values.astype(np.float32))
    metrics = compute_metrics(
        le.transform(test_df["family_id"]), y_pred,
        label_map={idx: name for idx, name in enumerate(le.classes_)},
        time_costs={"train_time": clf.train_time, "inference_time_total": clf.inference_time_total, "inference_time_per_sample": clf.inference_time_per_sample},
        save_dir=str(EXPR_DIR / tag), tag=tag,
    )
    return _save_metrics(tag, metrics, "A", model_type)


def run_group_b(old_df, new_df, families, model_type, window_size, step):
    tag = f"yourthings_family_group_b_{model_type}_W{window_size}_S{step}"
    train_feat = _build_windows_family(old_df, "train", families, window_size, step)
    test_feat = _build_windows_family(new_df, None, families, window_size, step)
    le = LabelEncoder().fit(families)
    feature_cols = [c for c in train_feat.columns if c != "family_id"]
    clf = _build_classifier(model_type)
    clf.train(train_feat[feature_cols].fillna(0.0).values.astype(np.float32), le.transform(train_feat["family_id"]))
    y_pred = clf.predict(test_feat[feature_cols].fillna(0.0).values.astype(np.float32))
    metrics = compute_metrics(
        le.transform(test_feat["family_id"]), y_pred,
        label_map={idx: name for idx, name in enumerate(le.classes_)},
        time_costs={"train_time": clf.train_time, "inference_time_total": clf.inference_time_total, "inference_time_per_sample": clf.inference_time_per_sample},
        save_dir=str(EXPR_DIR / tag), tag=tag,
    )
    return _save_metrics(tag, metrics, "B", model_type)


def run_group_c(old_df, new_df, families, model_type, seq_length, step):
    tag = f"yourthings_family_group_c_{model_type}_H{seq_length}_L{step}"
    train_feat, train_labels = _build_seq_compressed_family(old_df, "train", families, seq_length, step)
    test_feat, test_labels = _build_seq_compressed_family(new_df, None, families, seq_length, step)
    le = LabelEncoder().fit(families)
    clf = _build_classifier(model_type)
    clf.train(train_feat.values.astype(np.float32), le.transform(train_labels))
    y_pred = clf.predict(test_feat.values.astype(np.float32))
    metrics = compute_metrics(
        le.transform(test_labels), y_pred,
        label_map={idx: name for idx, name in enumerate(le.classes_)},
        time_costs={"train_time": clf.train_time, "inference_time_total": clf.inference_time_total, "inference_time_per_sample": clf.inference_time_per_sample},
        save_dir=str(EXPR_DIR / tag), tag=tag,
    )
    return _save_metrics(tag, metrics, "C", model_type)


def run_group_d(old_df, new_df, families, rnn_type, seq_length, step):
    tag = f"yourthings_family_group_d_{rnn_type}_H{seq_length}_L{step}"
    X_train, y_train_labels = _build_raw_sequences_family(old_df, "train", families, seq_length, step)
    X_val, y_val_labels = _build_raw_sequences_family(old_df, "val", families, seq_length, step)
    X_test, y_test_labels = _build_raw_sequences_family(new_df, None, families, seq_length, step)
    le = LabelEncoder().fit(families)
    y_train = le.transform(y_train_labels)
    y_val = le.transform(y_val_labels)
    y_test = le.transform(y_test_labels)
    scaler = StandardScaler()
    n_feat = X_train.shape[-1]
    X_train = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape).astype(np.float32)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape).astype(np.float32)
    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32)
    model = IoTRNNClassifier(input_size=len(GROUP_D_FEATURES), hidden_size=64, num_layers=2, num_classes=len(families), rnn_type=rnn_type, dropout=0.3)
    trainer = RNNTrainer(model, device=DEVICE)
    hist = trainer.train(train_loader, val_loader, save_path=str(MODEL_DIR / f"{tag}_best.pt"))
    t0 = time.perf_counter()
    _, _, y_true, y_pred = trainer.evaluate(test_loader, measure_latency=True)
    t1 = time.perf_counter()
    metrics = compute_metrics(
        y_true, y_pred,
        label_map={idx: name for idx, name in enumerate(le.classes_)},
        time_costs={"train_time": hist["train_time"], "inference_time_total": t1 - t0, "inference_time_per_sample": (t1 - t0) / max(len(y_true), 1)},
        save_dir=str(EXPR_DIR / tag), tag=tag,
    )
    return _save_metrics(tag, metrics, "D", rnn_type)


def run_group_e0(old_df, new_df, families, model_type, sample_step, hist_window):
    tag = f"yourthings_family_group_e0_{model_type}_S{sample_step}_W{hist_window}"
    train_feat = _build_group_e_family(old_df, "train", families, sample_step, hist_window)
    test_feat = _build_group_e_family(new_df, None, families, sample_step, hist_window)
    feature_cols = [c for c in train_feat.columns if c != "family_id"]
    train_feat[feature_cols] = train_feat[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    test_feat[feature_cols] = test_feat[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    le = LabelEncoder().fit(families)
    clf = _build_classifier(model_type)
    clf.train(train_feat[feature_cols].values.astype(np.float32), le.transform(train_feat["family_id"]))
    y_pred = clf.predict(test_feat[feature_cols].values.astype(np.float32))
    metrics = compute_metrics(
        le.transform(test_feat["family_id"]), y_pred,
        label_map={idx: name for idx, name in enumerate(le.classes_)},
        time_costs={"train_time": clf.train_time, "inference_time_total": clf.inference_time_total, "inference_time_per_sample": clf.inference_time_per_sample},
        save_dir=str(EXPR_DIR / tag), tag=tag,
    )
    return _save_metrics(tag, metrics, "E0", model_type)


def run_group_e1(old_df, new_df, families, model_type, sample_step, hist_window, d_max):
    tag = f"yourthings_family_group_e1_{model_type}_S{sample_step}_W{hist_window}"
    train_feat = _build_group_e1_family(old_df, "train", families, sample_step, hist_window, d_max)
    test_feat = _build_group_e1_family(new_df, None, families, sample_step, hist_window, d_max)
    feature_cols = [c for c in train_feat.columns if c != "family_id"]
    train_feat[feature_cols] = train_feat[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    test_feat[feature_cols] = test_feat[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    le = LabelEncoder().fit(families)
    clf = _build_classifier(model_type)
    clf.train(train_feat[feature_cols].values.astype(np.float32), le.transform(train_feat["family_id"]))
    y_pred = clf.predict(test_feat[feature_cols].values.astype(np.float32))
    metrics = compute_metrics(
        le.transform(test_feat["family_id"]), y_pred,
        label_map={idx: name for idx, name in enumerate(le.classes_)},
        time_costs={"train_time": clf.train_time, "inference_time_total": clf.inference_time_total, "inference_time_per_sample": clf.inference_time_per_sample},
        save_dir=str(EXPR_DIR / tag), tag=tag,
    )
    return _save_metrics(tag, metrics, "E1", model_type)


def run_all(max_packets_per_family, min_packets_per_family, max_files, groups, force_parse):
    csv_path = build_yourthings_family_csv(max_packets_per_family, min_packets_per_family, max_files, force_parse)
    old_df = _prepare_train_family_df()
    new_df = _load_test_family_df(csv_path)
    families = _common_families(old_df, new_df)
    logger.info("Common families (%d): %s", len(families), families)

    rows = []
    if "A" in groups:
        for model in ("rf", "xgb"):
            rows.append(run_group_a(old_df, new_df, families, model))
    if "B" in groups:
        for model in ("rf", "xgb"):
            rows.append(run_group_b(old_df, new_df, families, model, DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STEP))
    if "C" in groups:
        for model in ("rf", "xgb"):
            cfg = GROUP_C_CONFIG[model]
            rows.append(run_group_c(old_df, new_df, families, model, cfg["seq_length"], cfg["step"]))
    if "D" in groups:
        for model in ("lstm", "gru"):
            cfg = GROUP_D_CONFIG[model]
            rows.append(run_group_d(old_df, new_df, families, model, cfg["seq_length"], cfg["step"]))
    if "E0" in groups:
        for model in ("rf", "xgb"):
            rows.append(run_group_e0(old_df, new_df, families, model, DEFAULT_GROUP_E_SAMPLE_STEP, DEFAULT_GROUP_E_HIST_WINDOW))
    if "E1" in groups:
        for model in ("rf", "xgb"):
            rows.append(run_group_e1(old_df, new_df, families, model, DEFAULT_GROUP_E_SAMPLE_STEP, DEFAULT_GROUP_E_HIST_WINDOW, DEFAULT_GROUP_E1_D_MAX))

    df = pd.DataFrame(rows)
    out_path = TABLE_DIR / "yourthings_family_cross_summary.csv"
    df.to_csv(out_path, index=False)
    print(out_path)
    print(df.to_string(index=False))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_packets_per_family", type=int, default=50000)
    parser.add_argument("--min_packets_per_family", type=int, default=1000)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--groups", type=str, default="A,B,C,D,E0,E1")
    parser.add_argument("--force_parse", action="store_true")
    parser.add_argument("--build_only", action="store_true")
    args = parser.parse_args()

    csv_path = build_yourthings_family_csv(args.max_packets_per_family, args.min_packets_per_family, args.max_files, args.force_parse)
    if args.build_only:
        print(csv_path)
        return
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    run_all(args.max_packets_per_family, args.min_packets_per_family, args.max_files, groups, False)


if __name__ == "__main__":
    main()
