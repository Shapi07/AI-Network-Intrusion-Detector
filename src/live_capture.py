"""
live_capture.py
===============
Phase 12: Real-time network traffic capture and flow analysis for AINID.

This module provides real-time network traffic capture and statistical
flow aggregation capabilities using Scapy. It acts as an extension to
the existing AI Network Intrusion Detector (AINID) pipeline.

Architectural Flow
------------------
Real Network Traffic
  └─► Scapy Packet Capture (interface, duration, packet limit)
        └─► Bidirectional 5-Tuple Flow Aggregation
              └─► Statistical Feature Extraction (duration, bytes, ports, flags)
                    └─► Feature Validation & Compatibility Check
                          ├─► [Compatible] ──► Existing AINID Prediction Pipeline (src.predict)
                          └─► [Incompatible] ─► "LIVE MODEL INCOMPATIBLE" Diagnostic Report

Security & Scope Limitations
----------------------------
* DEFENSIVE MONITORING ONLY: Analyzes local network interface metadata.
* NO PAYLOAD INSPECTION: Strictly extracts header fields and packet counts.
* NO OFFENSIVE CAPABILITIES: Does not modify, inject, or forge network packets.
* SAFE CAPTURE BOUNDS: Default sessions limited to small windows (10s / 1000 pkts).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scapy.all import (
    ICMP,
    IP,
    IPv6,
    TCP,
    UDP,
    Packet,
    conf,
    get_if_list,
    sniff,
)

from src.config import (
    CLASS_NAMES,
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    SCALER_PATH,
    configure_logging,
)
from src.predict import load_prediction_artifacts, predict
from src.utils import timeit

logger = configure_logging(__name__)

# ──────────────────────────────────────────────────────────────
# Well-known port to service inference mapping
# ──────────────────────────────────────────────────────────────
WELL_KNOWN_PORTS: dict[int, str] = {
    80: "http",
    8080: "http",
    443: "http",  # Mapped to http/web for dataset compatibility
    53: "dns",
    25: "smtp",
    587: "smtp",
    465: "smtp",
    21: "ftp",
    20: "ftp",
    22: "ssh",
    23: "telnet",
    110: "pop3",
    143: "imap",
    123: "ntp",
    67: "dhcp",
    68: "dhcp",
}

# Features that cannot be reliably extracted from isolated live packet flows
# because they depend on testbed-wide historical state counters.
UNSUPPORTED_LIVE_FEATURES: frozenset[str] = frozenset(
    {
        "ct_srv_src",
        "ct_state_ttl",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "is_sqrk",
        "is_sm_ips_ports",
        "ct_flw_http_mthd",
        "is_ftp_login",
        "ct_ftp_cmd",
    }
)


# ──────────────────────────────────────────────────────────────
# Flow Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class FlowStats:
    """Statistical tracking container for a bidirectional 5-tuple flow."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: float
    end_time: float
    src_bytes: int = 0
    dst_bytes: int = 0
    src_pkts: int = 0
    dst_pkts: int = 0
    min_pkt_size: int = sys.maxsize
    max_pkt_size: int = 0
    tcp_syn_count: int = 0
    tcp_ack_count: int = 0
    tcp_fin_count: int = 0
    tcp_rst_count: int = 0
    icmp_echo_count: int = 0
    inferred_service: str = "private"

    def update(self, packet: Packet, direction_src: bool) -> None:
        """Update flow statistics with an incoming packet."""
        pkt_len = len(packet)
        pkt_time = float(packet.time)

        if self.start_time == 0.0 or pkt_time < self.start_time:
            self.start_time = pkt_time
        if pkt_time > self.end_time:
            self.end_time = pkt_time

        self.min_pkt_size = min(self.min_pkt_size, pkt_len)
        self.max_pkt_size = max(self.max_pkt_size, pkt_len)

        if direction_src:
            self.src_pkts += 1
            self.src_bytes += pkt_len
        else:
            self.dst_pkts += 1
            self.dst_bytes += pkt_len

        # Extract TCP control flags if TCP
        if packet.haslayer(TCP):
            flags = packet[TCP].flags
            if "S" in flags:
                self.tcp_syn_count += 1
            if "A" in flags:
                self.tcp_ack_count += 1
            if "F" in flags:
                self.tcp_fin_count += 1
            if "R" in flags:
                self.tcp_rst_count += 1

        # Extract ICMP type if ICMP
        if packet.haslayer(ICMP):
            icmp_type = packet[ICMP].type
            if icmp_type in (8, 0):  # Echo request or echo reply
                self.icmp_echo_count += 1

    @property
    def duration(self) -> float:
        """Flow duration in seconds (guaranteed >= 0.0)."""
        return max(0.0, self.end_time - self.start_time)

    @property
    def total_pkts(self) -> int:
        """Total packet count."""
        return self.src_pkts + self.dst_pkts

    @property
    def total_bytes(self) -> int:
        """Total byte count."""
        return self.src_bytes + self.dst_bytes

    @property
    def packets_per_sec(self) -> float:
        """Packet transmission rate."""
        dur = self.duration
        return (self.total_pkts / dur) if dur > 0 else float(self.total_pkts)

    @property
    def bytes_per_sec(self) -> float:
        """Byte transmission rate."""
        dur = self.duration
        return (self.total_bytes / dur) if dur > 0 else float(self.total_bytes)

    @property
    def avg_pkt_size(self) -> float:
        """Average packet size."""
        total = self.total_pkts
        return (self.total_bytes / total) if total > 0 else 0.0

    @property
    def derived_flag(self) -> str:
        """
        Derive dataset-compatible connection flag state from observed TCP flags.
        Returns 'SF' for normal establishment/data, 'S0' for unanswered SYN,
        'REJ' for connection reset, or 'OTH' for others.
        """
        if self.protocol != "tcp":
            return "SF"
        if self.tcp_rst_count > 0:
            return "REJ"
        if self.tcp_syn_count > 0 and self.tcp_ack_count == 0:
            return "S0"
        return "SF"


# ──────────────────────────────────────────────────────────────
# System & Environment Checks
# ──────────────────────────────────────────────────────────────

def is_admin() -> bool:
    """
    Check if the current process has Administrator privileges (Windows).
    """
    if sys.platform != "win32":
        return True
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001
        return False


def list_interfaces() -> list[dict[str, str]]:
    """
    List available network interfaces formatted for Windows/Cross-platform diagnostic display.

    Returns
    -------
    list[dict[str, str]]
        List of interface dicts with keys: 'name', 'description', 'ip'.
    """
    interfaces = []
    try:
        ifaces = conf.ifaces
        for key in ifaces:
            iface = ifaces[key]
            interfaces.append(
                {
                    "name": getattr(iface, "name", str(key)),
                    "description": getattr(iface, "description", getattr(iface, "name", str(key))),
                    "ip": getattr(iface, "ip", "N/A"),
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not query Scapy detailed interfaces: %s. Falling back to get_if_list().", exc)
        for iface_name in get_if_list():
            interfaces.append({"name": iface_name, "description": iface_name, "ip": "Unknown"})

    return interfaces


# ──────────────────────────────────────────────────────────────
# Flow Aggregation & Feature Extraction Logic
# ──────────────────────────────────────────────────────────────

def create_flow_key(packet: Packet) -> tuple[str, str, int, int, str] | None:
    """
    Generate a canonical bidirectional 5-tuple flow key from a Scapy packet.

    The key orders IPs and ports deterministically so both directions of
    a flow (A -> B and B -> A) map to the exact same flow key.

    Parameters
    ----------
    packet : Packet
        Scapy packet.

    Returns
    -------
    tuple[str, str, int, int, str] | None
        5-tuple key: (min_ip, max_ip, min_port, max_port, protocol_str),
        or None if the packet does not contain IP or supported transport layers.
    """
    if IP in packet:
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
    elif IPv6 in packet:
        ip_src = packet[IPv6].src
        ip_dst = packet[IPv6].dst
    else:
        return None

    proto = "other"
    src_port = 0
    dst_port = 0

    if TCP in packet:
        proto = "tcp"
        src_port = int(packet[TCP].sport)
        dst_port = int(packet[TCP].dport)
    elif UDP in packet:
        proto = "udp"
        src_port = int(packet[UDP].sport)
        dst_port = int(packet[UDP].dport)
    elif ICMP in packet:
        proto = "icmp"

    # Canonical bidirectional sorting
    if (ip_src, src_port) <= (ip_dst, dst_port):
        return (ip_src, ip_dst, src_port, dst_port, proto)
    return (ip_dst, ip_src, dst_port, src_port, proto)


def infer_service(src_port: int, dst_port: int, proto: str) -> str:
    """
    Infer likely network service from transport ports and protocol.

    Parameters
    ----------
    src_port : int
        Source port.
    dst_port : int
        Destination port.
    proto : str
        Protocol ('tcp', 'udp', 'icmp').

    Returns
    -------
    str
        Inferred service name (e.g. 'http', 'dns', 'smtp', 'ftp', 'eco_i', 'private').
    """
    if proto == "icmp":
        return "eco_i"
    if dst_port in WELL_KNOWN_PORTS:
        return WELL_KNOWN_PORTS[dst_port]
    if src_port in WELL_KNOWN_PORTS:
        return WELL_KNOWN_PORTS[src_port]
    return "private"


def aggregate_packets(packets: list[Packet]) -> list[FlowStats]:
    """
    Group raw Scapy packets into bidirectional flows and aggregate statistical metrics.

    Parameters
    ----------
    packets : list[Packet]
        List of captured Scapy packet objects.

    Returns
    -------
    list[FlowStats]
        List of aggregated flow statistics objects.
    """
    flows: dict[tuple[str, str, int, int, str], FlowStats] = {}

    for packet in packets:
        flow_key = create_flow_key(packet)
        if flow_key is None:
            continue

        ip_src = packet[IP].src if IP in packet else packet[IPv6].src
        src_port = packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else 0)
        dst_port = packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0)
        proto = flow_key[4]

        canonical_src, canonical_dst, canonical_sport, canonical_dport, _ = flow_key
        is_forward = (ip_src == canonical_src and src_port == canonical_sport)

        if flow_key not in flows:
            pkt_time = float(packet.time)
            service = infer_service(canonical_sport, canonical_dport, proto)
            flows[flow_key] = FlowStats(
                src_ip=canonical_src,
                dst_ip=canonical_dst,
                src_port=canonical_sport,
                dst_port=canonical_dport,
                protocol=proto,
                start_time=pkt_time,
                end_time=pkt_time,
                inferred_service=service,
            )

        flows[flow_key].update(packet, direction_src=is_forward)

    # Clean up minimum packet sizes for uninitialized flows
    for f in flows.values():
        if f.min_pkt_size == sys.maxsize:
            f.min_pkt_size = 0

    return list(flows.values())


def extract_flow_features(flow_list: list[FlowStats]) -> pd.DataFrame:
    """
    Convert aggregated flow statistics into a Pandas DataFrame compatible
    with feature engineering schemas.

    Parameters
    ----------
    flow_list : list[FlowStats]
        List of aggregated flow objects.

    Returns
    -------
    pd.DataFrame
        DataFrame containing both flow metadata and statistical features.
    """
    if not flow_list:
        return pd.DataFrame()

    records = []
    for f in flow_list:
        records.append(
            {
                # Metadata columns
                "src_ip": f.src_ip,
                "dst_ip": f.dst_ip,
                "src_port": f.src_port,
                "dst_port": f.dst_port,
                # Primary dataset features
                "duration": round(f.duration, 4),
                "src_bytes": f.src_bytes,
                "dst_bytes": f.dst_bytes,
                "protocol_type": f.protocol,
                "service": f.inferred_service,
                "flag": f.derived_flag,
                "spkts": f.src_pkts,
                "dpkts": f.dst_pkts,
                "total_pkts": f.total_pkts,
                "total_bytes": f.total_bytes,
                "packets_per_sec": round(f.packets_per_sec, 2),
                "bytes_per_sec": round(f.bytes_per_sec, 2),
                "avg_pkt_size": round(f.avg_pkt_size, 2),
            }
        )

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# Machine Learning Feature Compatibility Verification
# ──────────────────────────────────────────────────────────────

def validate_live_features(
    expected_feature_names: list[str], df_live_features: pd.DataFrame
) -> tuple[bool, list[str], str]:
    """
    Validate whether the trained model's required features can be legitimately
    derived from live network traffic.

    CRITICAL SAFETY RULE:
    A missing feature is NOT automatically zero. If the trained model requires
    features that cannot be reliably extracted from live traffic (such as host multi-flow
    lab counters like 'ct_srv_src'), this function marks the model as INCOMPATIBLE
    to prevent fake ML predictions.

    Parameters
    ----------
    expected_feature_names : list[str]
        Feature names expected by the trained model (from feature_names.joblib).
    df_live_features : pd.DataFrame
        Extracted live feature DataFrame.

    Returns
    -------
    tuple[bool, list[str], str]
        * is_compatible (True if model can safely predict, False otherwise).
        * list of incompatible feature names found.
        * technical reason message.
    """
    incompatible = [f for f in expected_feature_names if f in UNSUPPORTED_LIVE_FEATURES]

    if incompatible:
        msg = (
            f"Model requires {len(incompatible)} feature(s) that cannot be "
            f"reliably extracted from live traffic: {incompatible}. "
            "Generating predictions with zero-filled fake values is disabled "
            "for scientific accuracy."
        )
        logger.error("❌ Live ML Compatibility Check Failed: %s", msg)
        return False, incompatible, msg

    logger.info("✅ Live ML Compatibility Check Passed: Model features compatible with live traffic.")
    return True, [], "Compatible"


# ──────────────────────────────────────────────────────────────
# Packet Capture Engine
# ──────────────────────────────────────────────────────────────

@timeit
def capture_packets(
    interface: str | None = None,
    duration: int = 10,
    max_packets: int = 1000,
) -> list[Packet]:
    """
    Capture live network packets using Scapy with safe session bounds.

    Parameters
    ----------
    interface : str, optional
        Name of the network interface. Defaults to None (Scapy auto-selects default).
    duration : int, optional
        Capture duration in seconds. Defaults to 10.
    max_packets : int, optional
        Maximum number of packets to sniff. Defaults to 1000.

    Returns
    -------
    list[Packet]
        List of captured Scapy packet objects.

    Raises
    ------
    PermissionError
        If Administrator privileges are missing on Windows.
    RuntimeError
        If packet capture fails or interface cannot be opened.
    """
    if sys.platform == "win32" and not is_admin():
        logger.warning("⚠️ Running on Windows without Administrator privileges. Packet capture may fail.")

    logger.info(
        "🎧 Starting packet capture on interface '%s' (timeout=%ds, limit=%d pkts)...",
        interface or "default",
        duration,
        max_packets,
    )

    try:
        kwargs: dict[str, Any] = {
            "timeout": duration,
            "count": max_packets,
            "store": True,
        }
        if interface and interface.strip():
            kwargs["iface"] = interface.strip()

        packets = sniff(**kwargs)
        logger.info("✅ Capture complete: %d packet(s) captured.", len(packets))
        return list(packets)

    except PermissionError as exc:
        raise PermissionError(
            "Insufficient privileges to open network interface. "
            "Please run command prompt / terminal as Administrator."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Packet capture error on interface '{interface}': {exc}. "
            "Ensure Npcap is installed on Windows and the interface name is valid."
        ) from exc


# ──────────────────────────────────────────────────────────────
# Top-Level Detection Orchestrator
# ──────────────────────────────────────────────────────────────

@timeit
def run_live_detection(
    interface: str | None = None,
    duration: int = 10,
    max_packets: int = 1000,
) -> dict[str, Any]:
    """
    Top-level orchestrator for Phase 12 real-time network traffic analysis.

    Sequence:
    1. Capture packets (Scapy)
    2. Aggregate packets into 5-tuple flows
    3. Extract flow features into DataFrame
    4. Load trained model artifacts & validate feature compatibility
    5. Run inference via src.predict (if compatible) or return diagnostic report

    Parameters
    ----------
    interface : str, optional
        Network interface name.
    duration : int, optional
        Capture duration in seconds.
    max_packets : int, optional
        Maximum packet limit.

    Returns
    -------
    dict[str, Any]
        Dictionary containing capture metrics, flows, compatibility status,
        and predictions dataframe (or diagnostic message).
    """
    packets = capture_packets(interface=interface, duration=duration, max_packets=max_packets)

    if not packets:
        return {
            "status": "EMPTY_CAPTURE",
            "packets_captured": 0,
            "flows_detected": 0,
            "message": "No network packets were captured within the specified timeout.",
        }

    # Flow aggregation
    flows = aggregate_packets(packets)
    df_live = extract_flow_features(flows)

    # Check model artifacts
    try:
        artifacts = load_prediction_artifacts()
    except FileNotFoundError as exc:
        return {
            "status": "MISSING_MODEL",
            "packets_captured": len(packets),
            "flows_detected": len(flows),
            "message": f"Trained model artifacts missing: {exc}. Please run model training first.",
        }

    # Validate feature compatibility
    is_compatible, incompatible_cols, reason = validate_live_features(
        artifacts.feature_names, df_live
    )

    if not is_compatible:
        return {
            "status": "LIVE_MODEL_INCOMPATIBLE",
            "packets_captured": len(packets),
            "flows_detected": len(flows),
            "incompatible_features": incompatible_cols,
            "reason": reason,
            "recommendation": "Train a live-compatible model using standard flow statistical features.",
            "df_flows": df_live,
        }

    # Run inference reusing src.predict.predict
    predictions_df = predict(df_live, artifacts=artifacts, save_output=False)

    n_attack = int((predictions_df["prediction"] == 1).sum())
    n_normal = len(predictions_df) - n_attack

    return {
        "status": "SUCCESS",
        "packets_captured": len(packets),
        "flows_detected": len(flows),
        "n_normal": n_normal,
        "n_attack": n_attack,
        "predictions": predictions_df,
        "df_flows": df_live,
    }


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AINID Phase 12 — Real-Time Network Traffic Monitor."
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List available network capture interfaces on this system.",
    )
    parser.add_argument(
        "-i",
        "--interface",
        type=str,
        default=None,
        help="Network interface to capture traffic from (default: auto-select).",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=10,
        help="Capture duration timeout in seconds (default: 10).",
    )
    parser.add_argument(
        "-m",
        "--max-packets",
        type=int,
        default=1000,
        help="Maximum packet count to capture (default: 1000).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.list_interfaces:
        print("\n========================================")
        print("AINID AVAILABLE NETWORK INTERFACES")
        print("========================================")
        ifaces = list_interfaces()
        for idx, iface in enumerate(ifaces, 1):
            print(f"{idx:2d}. Name: {iface['name']}")
            print(f"    Description: {iface['description']}")
            print(f"    IP: {iface['ip']}")
            print("-" * 40)
        return

    print("\n========================================")
    print("AINID LIVE NETWORK MONITOR")
    print("========================================")
    print(f"Interface: {args.interface or 'Auto-Select (Default)'}")
    print(f"Duration:  {args.duration} seconds")
    print(f"Max Pkts:  {args.max_packets}")
    print("----------------------------------------")

    try:
        result = run_live_detection(
            interface=args.interface,
            duration=args.duration,
            max_packets=args.max_packets,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ CAPTURE ERROR: {exc}")
        sys.exit(1)

    status = result["status"]

    if status == "EMPTY_CAPTURE":
        print("\n⚠️ No network packets captured within timeout.")
        print(result["message"])

    elif status == "MISSING_MODEL":
        print("\n❌ MODEL ERROR:")
        print(result["message"])

    elif status == "LIVE_MODEL_INCOMPATIBLE":
        print("\n========================================")
        print("AINID LIVE NETWORK MONITOR")
        print("========================================")
        print("Prediction unavailable.")
        print("\nReason:")
        print(result["reason"])
        print("\nRecommendation:")
        print(result["recommendation"])
        print("========================================")

    elif status == "SUCCESS":
        print(f"\nPackets captured: {result['packets_captured']}")
        print(f"Flows detected:   {result['flows_detected']}")
        print(f"\nNormal:           {result['n_normal']}")
        print(f"Potential attacks: {result['n_attack']}")

        predictions = result["predictions"]
        print("\n" + "=" * 50)
        print("FLOW PREDICTION BREAKDOWN")
        print("=" * 50)
        for idx, row in predictions.iterrows():
            flow_info = result["df_flows"].loc[idx]
            proto = str(flow_info["protocol_type"]).upper()
            pkts = flow_info["total_pkts"]
            bytes_cnt = flow_info["total_bytes"]
            dur = flow_info["duration"]
            pred_lbl = row["prediction_label"]
            prob = row.get("attack_probability", "N/A")
            prob_str = f"{prob * 100:.1f}%" if isinstance(prob, (float, int)) else "N/A"

            print(f"Flow {idx + 1}: {flow_info['src_ip']}:{flow_info['src_port']} -> {flow_info['dst_ip']}:{flow_info['dst_port']}")
            print(f"  Protocol: {proto} | Packets: {pkts} | Bytes: {bytes_cnt} | Duration: {dur:.2f}s")
            print(f"  Verdict:  {pred_lbl} (Attack Prob: {prob_str})")
            print("-" * 50)

        print("========================================")


if __name__ == "__main__":
    main()
