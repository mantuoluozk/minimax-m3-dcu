#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy/common.sh"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniMax-M3}"

SERVER_PORT=${SERVER_PORT:-40000}
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
DEEPEP_CONFIG="${DEEPEP_CONFIG:-$SCRIPT_DIR/deploy/config/deepep-prefill.json}"
for config in "$DEEPEP_CONFIG" "${DEEPGEMM_CONFIG:-$SCRIPT_DIR/deploy/config/deepgemm-e16.json}"; do
    if [[ ! -r "$config" ]]; then
        printf 'Cannot read Prefill config: %s\n' "$config" >&2
        exit 1
    fi
done
export PYTHONPATH="$PROJECT_ROOT/sglang-das/python:$PROJECT_ROOT/build"
export SGLANG_ENABLE_CP_V2=1
export SGLANG_MINIMAX_CP_FP8_KV_EXCHANGE=1
export SGLANG_HEALTH_CHECK_TIMEOUT=120
export SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=1

export NCCL_IB_HCA="${NCCL_IB_HCA:-${IB_DEVICES//,/:1,}:1}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="$IB_DEVICES"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="$IB_DEVICES"
export MC_ALLOWED_IBV_DEVICES="$IB_DEVICES"

export SGLANG_HCU_DEEPGEMM_CONTIG_TUNING_CONFIG="${DEEPGEMM_CONFIG:-$SCRIPT_DIR/deploy/config/deepgemm-e16.json}"

unset SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER SGLANG_USE_FUSED_TOPK_SOFTMAX
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_KV_LAYOUT_HCU_FA=false
export SGLANG_REQUIRE_KV_CACHE_SCALES=1
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938=0
export SGLANG_OPT_USE_BF16_ROUTER_GEMM=1
export SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=0
export SGLANG_OPT_USE_MINIMAX_QUERY_SHARDED_CP=1
export SGLANG_PREFILL_CP_MIN_TOKENS_PER_SEQUENCE=4096
export SGLANG_MINIMAX_PARALLEL_CP_SEGMENTS=1
export SGLANG_MINIMAX_PREFILL_TOPK_BACKEND=aiter
export SGLANG_OPT_USE_MINIMAX_DEEPEP_SHARED_EXPERT_OVERLAP=1
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_DEEPEP_ASYNC_FINISH=0
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_SET_CPU_AFFINITY=1
export NCCL_MIN_NCHANNELS=48
export NCCL_MAX_NCHANNELS=48
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/prefill-$(date +%Y%m%d-%H%M%S)-$$.log"
printf "Log file: %s\n" "$LOG_FILE"

python3 -u -m sglang.launch_server \
  --dist-init-addr "${DIST_INIT_ADDR:-127.0.0.1:26000}" \
  --enable-dp-attention --disable-overlap-schedule \
  --moe-dense-tp-size 1 --prefill-max-requests 4 \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --disaggregation-ib-device "$IB_DEVICES" \
  --tp-size 8 \
  --ep-size 8 \
  --moe-dp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "$DEEPEP_CONFIG" \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --ep-dispatch-algorithm static \
  --disable-shared-experts-fusion \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --attention-context-parallel-size 8 \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --context-length 163840 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 32 \
  --mem-fraction-static 0.80 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --disable-radix-cache \
  --disable-custom-all-reduce 2>&1 | tee "$LOG_FILE"
