#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH inside the container}"
: "${NETWORK_IFACE:?Set NETWORK_IFACE to the host interface}"
: "${IB_DEVICES:?Set IB_DEVICES to a comma-separated RDMA device list}"
LOG_DIR="${LOG_DIR:-/tmp/minimax-m3-logs}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$NETWORK_IFACE}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NETWORK_IFACE}"
