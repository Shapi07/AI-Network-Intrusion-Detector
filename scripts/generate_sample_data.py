"""
generate_sample_data.py
=======================
Generate a small synthetic UNSW-NB15-like CSV for development and testing.

This script creates a realistic-looking dataset with the same column
structure as UNSW-NB15 (49 features + label), but with randomly generated
values.  It is NOT real network traffic — it is only useful for:

  * Testing the preprocessing pipeline without downloading a 100 MB dataset.
  * Running the Streamlit UI in demo mode.
  * CI / unit tests in GitHub Actions.

Usage
-----
  python scripts/generate_sample_data.py

Output
------
  data/raw/sample_unsw_nb15.csv  (5 000 rows)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── make sure project root is on sys.path ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR  # noqa: E402

RANDOM_SEED = 42
N_ROWS = 5_000
OUTPUT_PATH = RAW_DATA_DIR / "sample_unsw_nb15.csv"

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


def _rand_ip() -> str:
    return ".".join(str(rng.integers(1, 255)) for _ in range(4))


def generate() -> pd.DataFrame:
    """
    Build a synthetic UNSW-NB15-style dataframe.

    Column groups
    -------------
    * Network flow identifiers (src/dst IP, ports, protocol)
    * Flow statistics (duration, bytes, packets)
    * Content-level features (payload entropy, TTL)
    * Labels (binary: 0 = normal, 1 = attack)
    """
    protos   = ["tcp", "udp", "icmp", "arp", "ospf"]
    services = ["-", "http", "ftp", "smtp", "ssh", "dns", "ssl", "dhcp"]
    states   = ["FIN", "INT", "CON", "REQ", "RST", "PAR", "URN", "no"]

    n = N_ROWS

    # ── identifiers (will be dropped by preprocessing) ──────────
    id_col       = np.arange(1, n + 1)
    srcip        = [_rand_ip() for _ in range(n)]
    sport        = rng.integers(1024, 65535, n)
    dstip        = [_rand_ip() for _ in range(n)]
    dsport       = rng.integers(1, 1024, n)

    # ── categorical ──────────────────────────────────────────────
    proto_col    = rng.choice(protos, n)
    service_col  = rng.choice(services, n)
    state_col    = rng.choice(states, n)

    # ── flow stats ───────────────────────────────────────────────
    dur          = rng.exponential(scale=0.5,  size=n).round(6)
    sbytes       = rng.integers(0, 1_000_000, n)
    dbytes       = rng.integers(0, 500_000,   n)
    sttl         = rng.integers(1, 255, n)
    dttl         = rng.integers(1, 255, n)
    sloss        = rng.integers(0, 50,  n)
    dloss        = rng.integers(0, 50,  n)
    sload        = rng.uniform(0, 1e6, n).round(4)
    dload        = rng.uniform(0, 1e6, n).round(4)
    spkts        = rng.integers(1, 500, n)
    dpkts        = rng.integers(1, 500, n)
    swin         = rng.integers(0, 65535, n)
    dwin         = rng.integers(0, 65535, n)
    stcpb        = rng.integers(0, 2**31, n)
    dtcpb        = rng.integers(0, 2**31, n)
    smeansz      = rng.integers(20, 1500, n)
    dmeansz      = rng.integers(20, 1500, n)
    trans_depth  = rng.integers(0, 10, n)
    res_bdy_len  = rng.integers(0, 50000, n)
    sjit         = rng.uniform(0, 1, n).round(6)
    djit         = rng.uniform(0, 1, n).round(6)
    stime        = rng.integers(1_400_000_000, 1_600_000_000, n)
    ltime        = stime + rng.integers(0, 3600, n)
    sintpkt      = rng.uniform(0, 100, n).round(6)
    dintpkt      = rng.uniform(0, 100, n).round(6)
    tcprtt       = rng.uniform(0, 1, n).round(6)
    synack       = rng.uniform(0, 1, n).round(6)
    ackdat       = rng.uniform(0, 1, n).round(6)
    is_sm_ips_ports = rng.integers(0, 2, n)
    ct_state_ttl    = rng.integers(0, 6, n)
    ct_flw_http_mthd= rng.integers(0, 3, n)
    is_ftp_login    = rng.integers(0, 2, n)
    ct_ftp_cmd      = rng.integers(0, 5, n)
    ct_srv_src      = rng.integers(1, 100, n)
    ct_srv_dst      = rng.integers(1, 100, n)
    ct_dst_ltm      = rng.integers(1, 100, n)
    ct_src_ltm      = rng.integers(1, 100, n)
    ct_src_dport_ltm= rng.integers(1, 100, n)
    ct_dst_sport_ltm= rng.integers(1, 100, n)
    ct_dst_src_ltm  = rng.integers(1, 100, n)

    # ── label: ~40 % attack, 60 % normal ───────────────────────
    label       = rng.choice([0, 1], n, p=[0.60, 0.40])
    attack_cat  = [
        random.choice(["Normal", "Fuzzers", "Backdoor", "DoS",
                        "Exploits", "Generic", "Reconnaissance",
                        "Shellcode", "Worms"])
        if lbl == 1 else "Normal"
        for lbl in label
    ]

    df = pd.DataFrame({
        "id":               id_col,
        "dur":              dur,
        "proto":            proto_col,
        "service":          service_col,
        "state":            state_col,
        "spkts":            spkts,
        "dpkts":            dpkts,
        "sbytes":           sbytes,
        "dbytes":           dbytes,
        "sttl":             sttl,
        "dttl":             dttl,
        "sloss":            sloss,
        "dloss":            dloss,
        "sload":            sload,
        "dload":            dload,
        "sinpkt":           sintpkt,
        "dinpkt":           dintpkt,
        "sjit":             sjit,
        "djit":             djit,
        "swin":             swin,
        "stcpb":            stcpb,
        "dtcpb":            dtcpb,
        "dwin":             dwin,
        "tcprtt":           tcprtt,
        "synack":           synack,
        "ackdat":           ackdat,
        "smean":            smeansz,
        "dmean":            dmeansz,
        "trans_depth":      trans_depth,
        "response_body_len":res_bdy_len,
        "ct_srv_src":       ct_srv_src,
        "ct_state_ttl":     ct_state_ttl,
        "ct_dst_ltm":       ct_dst_ltm,
        "ct_src_dport_ltm": ct_src_dport_ltm,
        "ct_dst_sport_ltm": ct_dst_sport_ltm,
        "ct_dst_src_ltm":   ct_dst_src_ltm,
        "is_ftp_login":     is_ftp_login,
        "ct_ftp_cmd":       ct_ftp_cmd,
        "ct_flw_http_mthd": ct_flw_http_mthd,
        "ct_src_ltm":       ct_src_ltm,
        "ct_srv_dst":       ct_srv_dst,
        "is_sm_ips_ports":  is_sm_ips_ports,
        "attack_cat":       attack_cat,
        "label":            label,
    })

    return df


if __name__ == "__main__":
    print(f"Generating {N_ROWS:,} synthetic UNSW-NB15 rows …")
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved → {OUTPUT_PATH}")
    print(f"   Shape : {df.shape}")
    print(f"   Labels: {df['label'].value_counts().to_dict()}")
