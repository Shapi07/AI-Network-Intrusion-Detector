"""
test_live_capture.py
====================
Unit and integration tests for Phase 12 (src/live_capture.py).

All tests use synthetic/mocked Scapy packet objects and mocked model
artifacts — no real network interface or root privileges are required to
run this test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from scapy.all import ICMP, IP, TCP, UDP, Ether, Raw

from src.live_capture import (
    FlowStats,
    aggregate_packets,
    create_flow_key,
    extract_flow_features,
    infer_service,
    list_interfaces,
    run_live_detection,
    validate_live_features,
)


# ──────────────────────────────────────────────────────────────
# Helper: Create synthetic Scapy packets
# ──────────────────────────────────────────────────────────────

def make_tcp_packet(
    src_ip: str = "192.168.1.10",
    dst_ip: str = "10.0.0.1",
    sport: int = 49152,
    dport: int = 80,
    flags: str = "S",
    payload_len: int = 100,
    pkt_time: float = 1000.0,
) -> Ether:
    """Construct a synthetic Scapy TCP packet for testing."""
    pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags=flags)
    if payload_len > 0:
        pkt = pkt / Raw(load=b"X" * payload_len)
    pkt.time = pkt_time
    return pkt


def make_udp_packet(
    src_ip: str = "192.168.1.10",
    dst_ip: str = "8.8.8.8",
    sport: int = 5353,
    dport: int = 53,
    payload_len: int = 50,
    pkt_time: float = 1000.0,
) -> Ether:
    """Construct a synthetic Scapy UDP packet for testing."""
    pkt = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport)
    if payload_len > 0:
        pkt = pkt / Raw(load=b"U" * payload_len)
    pkt.time = pkt_time
    return pkt


def make_icmp_packet(
    src_ip: str = "192.168.1.10",
    dst_ip: str = "10.0.0.1",
    icmp_type: int = 8,
    pkt_time: float = 1000.0,
) -> Ether:
    """Construct a synthetic Scapy ICMP packet for testing."""
    pkt = Ether() / IP(src=src_ip, dst=dst_ip) / ICMP(type=icmp_type)
    pkt.time = pkt_time
    return pkt


# ──────────────────────────────────────────────────────────────
# 1. Flow Key & Directionality Tests
# ──────────────────────────────────────────────────────────────

def test_create_flow_key_bidirectional():
    """Verify A -> B and B -> A generate the exact same canonical flow key."""
    pkt1 = make_tcp_packet(src_ip="192.168.1.10", dst_ip="10.0.0.1", sport=5000, dport=80)
    pkt2 = make_tcp_packet(src_ip="10.0.0.1", dst_ip="192.168.1.10", sport=80, dport=5000)

    key1 = create_flow_key(pkt1)
    key2 = create_flow_key(pkt2)

    assert key1 is not None
    assert key2 is not None
    assert key1 == key2
    assert key1 == ("10.0.0.1", "192.168.1.10", 80, 5000, "tcp")


def test_create_flow_key_non_ip():
    """Verify non-IP packets return None for flow key."""
    pkt = Ether() / Raw(load=b"Non-IP Ethernet Frame")
    assert create_flow_key(pkt) is None


# ──────────────────────────────────────────────────────────────
# 2. Flow Aggregation & Statistics Tests
# ──────────────────────────────────────────────────────────────

def test_aggregate_packets_tcp_flow():
    """Test packet aggregation, byte counting, packet counting, and direction tracking."""
    # Forward packet (A -> B)
    pkt1 = make_tcp_packet(src_ip="192.168.1.10", dst_ip="10.0.0.1", sport=5000, dport=80, flags="S", pkt_time=100.0)
    # Reverse packet (B -> A)
    pkt2 = make_tcp_packet(src_ip="10.0.0.1", dst_ip="192.168.1.10", sport=80, dport=5000, flags="SA", pkt_time=102.5)

    flows = aggregate_packets([pkt1, pkt2])

    assert len(flows) == 1
    flow = flows[0]

    assert flow.total_pkts == 2
    assert flow.src_pkts == 1
    assert flow.dst_pkts == 1
    assert flow.duration == pytest.approx(2.5, abs=1e-3)
    assert flow.tcp_syn_count == 2  # S and SA both contain 'S'
    assert flow.tcp_ack_count == 1  # SA contains 'A'
    assert flow.derived_flag == "SF"
    assert flow.inferred_service == "http"


def test_aggregate_packets_udp():
    """Test UDP packet aggregation and service inference."""
    pkt = make_udp_packet(src_ip="192.168.1.10", dst_ip="8.8.8.8", sport=50000, dport=53, pkt_time=50.0)

    flows = aggregate_packets([pkt])

    assert len(flows) == 1
    flow = flows[0]

    assert flow.protocol == "udp"
    assert flow.inferred_service == "dns"
    assert flow.total_pkts == 1


def test_aggregate_packets_icmp():
    """Test ICMP packet aggregation."""
    pkt = make_icmp_packet(src_ip="192.168.1.10", dst_ip="10.0.0.1", icmp_type=8, pkt_time=10.0)

    flows = aggregate_packets([pkt])

    assert len(flows) == 1
    flow = flows[0]

    assert flow.protocol == "icmp"
    assert flow.inferred_service == "eco_i"
    assert flow.icmp_echo_count == 1


def test_zero_duration_flow():
    """Verify zero-duration single packet flows compute statistics without ZeroDivisionError."""
    pkt = make_tcp_packet(pkt_time=100.0)

    flows = aggregate_packets([pkt])
    assert len(flows) == 1
    flow = flows[0]

    assert flow.duration == 0.0
    assert flow.packets_per_sec == 1.0
    assert flow.bytes_per_sec == float(flow.total_bytes)
    assert flow.avg_pkt_size == float(flow.total_bytes)


def test_tcp_flags_s0_and_rej():
    """Test TCP flag state derivation for S0 (SYN unanswered) and REJ (reset)."""
    # S0 case: SYN only
    syn_pkt = make_tcp_packet(flags="S")
    flow_s0 = aggregate_packets([syn_pkt])[0]
    assert flow_s0.derived_flag == "S0"

    # REJ case: RST packet
    rst_pkt = make_tcp_packet(flags="R")
    flow_rej = aggregate_packets([rst_pkt])[0]
    assert flow_rej.derived_flag == "REJ"


# ──────────────────────────────────────────────────────────────
# 3. Service Detection Tests
# ──────────────────────────────────────────────────────────────

def test_infer_service_mappings():
    """Test port-to-service mapping helper."""
    assert infer_service(49152, 80, "tcp") == "http"
    assert infer_service(49152, 53, "udp") == "dns"
    assert infer_service(49152, 25, "tcp") == "smtp"
    assert infer_service(49152, 21, "tcp") == "ftp"
    assert infer_service(0, 0, "icmp") == "eco_i"
    assert infer_service(54321, 12345, "tcp") == "private"


# ──────────────────────────────────────────────────────────────
# 4. DataFrame Generation Tests
# ──────────────────────────────────────────────────────────────

def test_extract_flow_features_empty():
    """Verify empty flow list returns empty DataFrame."""
    df = extract_flow_features([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_extract_flow_features_shape():
    """Verify feature extraction creates correctly populated DataFrame."""
    pkt = make_tcp_packet()
    flows = aggregate_packets([pkt])
    df = extract_flow_features(flows)

    assert len(df) == 1
    assert "duration" in df.columns
    assert "src_bytes" in df.columns
    assert "dst_bytes" in df.columns
    assert "protocol_type" in df.columns
    assert "service" in df.columns
    assert "flag" in df.columns


# ──────────────────────────────────────────────────────────────
# 5. ML Compatibility Validation Tests (CRITICAL REQUIREMENT)
# ──────────────────────────────────────────────────────────────

def test_validate_live_features_compatible():
    """Test feature validation when model expected features are fully compatible."""
    expected_features = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "protocol_type_tcp",
        "protocol_type_udp",
        "service_http",
        "flag_SF",
    ]
    df_live = pd.DataFrame()

    is_compatible, incompatible, reason = validate_live_features(expected_features, df_live)

    assert is_compatible is True
    assert incompatible == []
    assert reason == "Compatible"


def test_validate_live_features_incompatible():
    """
    CRITICAL ML COMPATIBILITY TEST:
    Verify that if the model requires features that cannot be extracted
    from live traffic (e.g. ct_srv_src), validation REJECTS prediction
    and reports LIVE MODEL INCOMPATIBLE.
    """
    expected_features = [
        "duration",
        "src_bytes",
        "ct_srv_src",        # Unsupported live testbed feature
        "ct_dst_ltm",        # Unsupported live testbed feature
        "protocol_type_tcp",
    ]
    df_live = pd.DataFrame()

    is_compatible, incompatible, reason = validate_live_features(expected_features, df_live)

    assert is_compatible is False
    assert "ct_srv_src" in incompatible
    assert "ct_dst_ltm" in incompatible
    assert "cannot be reliably extracted" in reason


# ──────────────────────────────────────────────────────────────
# 6. System Utility Tests
# ──────────────────────────────────────────────────────────────

def test_list_interfaces():
    """Test interface enumeration function does not raise errors."""
    ifaces = list_interfaces()
    assert isinstance(ifaces, list)
    if ifaces:
        assert "name" in ifaces[0]


# ──────────────────────────────────────────────────────────────
# 7. Integration & Top-Level Detection Tests
# ──────────────────────────────────────────────────────────────

@patch("src.live_capture.sniff")
def test_run_live_detection_empty(mock_sniff):
    """Test top-level orchestrator handling empty capture."""
    mock_sniff.return_value = []

    res = run_live_detection(duration=1)

    assert res["status"] == "EMPTY_CAPTURE"
    assert res["packets_captured"] == 0
    assert res["flows_detected"] == 0


@patch("src.live_capture.load_prediction_artifacts")
@patch("src.live_capture.sniff")
def test_run_live_detection_incompatible_model(mock_sniff, mock_load_artifacts):
    """Test top-level orchestrator rejecting incompatible model."""
    mock_sniff.return_value = [make_tcp_packet()]

    # Mock artifacts requiring unsupported feature
    mock_artifacts = MagicMock()
    mock_artifacts.feature_names = ["duration", "ct_srv_src"]
    mock_load_artifacts.return_value = mock_artifacts

    res = run_live_detection(duration=1)

    assert res["status"] == "LIVE_MODEL_INCOMPATIBLE"
    assert res["packets_captured"] == 1
    assert res["flows_detected"] == 1
    assert "incompatible_features" in res
    assert "ct_srv_src" in res["incompatible_features"]


@patch("src.live_capture.predict")
@patch("src.live_capture.load_prediction_artifacts")
@patch("src.live_capture.sniff")
def test_run_live_detection_success(mock_sniff, mock_load_artifacts, mock_predict):
    """Test full successful live detection flow with compatible model."""
    mock_sniff.return_value = [make_tcp_packet()]

    # Mock artifacts with compatible features
    mock_artifacts = MagicMock()
    mock_artifacts.feature_names = ["duration", "src_bytes", "protocol_type_tcp"]
    mock_load_artifacts.return_value = mock_artifacts

    # Mock predictions output
    mock_pred_df = pd.DataFrame(
        {
            "prediction": [0],
            "prediction_label": ["Normal"],
            "attack_probability": [0.05],
        }
    )
    mock_predict.return_value = mock_pred_df

    res = run_live_detection(duration=1)

    assert res["status"] == "SUCCESS"
    assert res["packets_captured"] == 1
    assert res["flows_detected"] == 1
    assert res["n_normal"] == 1
    assert res["n_attack"] == 0
    assert "predictions" in res
