#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/deploy/common.sh"
NODE_RANK=0
SERVER_PORT=${SERVER_PORT:-40000}
export SGLANG_M31_AUDIT_DEEPGEMM="${SGLANG_M31_AUDIT_DEEPGEMM:-0}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
DEEPEP_CONFIG="${DEEPEP_CONFIG:-$PROJECT_ROOT/deploy/config/deepep-prefill.json}"
export PYTHONPATH="$PROJECT_ROOT/sglang-das/python:$PROJECT_ROOT/build"
export SGLANG_ENABLE_CP_V2=1
export SGLANG_MINIMAX_CP_FP8_KV_EXCHANGE=1
export SGLANG_HEALTH_CHECK_TIMEOUT=120
export SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=1



mkdir -p "$LOG_DIR"

export ROCM_HOME=/opt/dtk
export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk/hip



export NCCL_IB_HCA="${NCCL_IB_HCA:-${IB_DEVICES//,/:1,}:1}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${IB_DEVICES}"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="${IB_DEVICES}"
export MC_ALLOWED_IBV_DEVICES="${IB_DEVICES}"



export SGLANG_HCU_DEEPGEMM_CONTIG_TUNING_CONFIG="${DEEPGEMM_CONFIG:-$PROJECT_ROOT/deploy/config/deepgemm-e16.json}"



unset SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER SGLANG_USE_FUSED_TOPK_SOFTMAX
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_KV_LAYOUT_HCU_FA=false
export SGLANG_REQUIRE_KV_CACHE_SCALES=1
export SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=0
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938=0
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_INDEXER=0
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_DECODE=0
export SGLANG_OPT_USE_MINIMAX_QUERY_SHARDED_CP=1
export SGLANG_PREFILL_CP_MIN_TOKENS_PER_SEQUENCE=4096
export SGLANG_MINIMAX_COMBINE_CP_SEGMENTS=0
export SGLANG_MINIMAX_PARALLEL_CP_SEGMENTS=1
export SGLANG_MINIMAX_PREFILL_TOPK_BACKEND=aiter
export SGLANG_OPT_USE_MINIMAX_DEEPEP_SHARED_EXPERT_OVERLAP=1
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_DEEPEP_ASYNC_FINISH=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_OPT_USE_BF16_ROUTER_GEMM=1
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_SET_CPU_AFFINITY=1
export NCCL_MIN_NCHANNELS=48
export NCCL_MAX_NCHANNELS=48
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600

python3 -m sglang.launch_server --dist-init-addr "${DIST_INIT_ADDR:-127.0.0.1:26000}" --enable-dp-attention --disable-overlap-schedule --moe-dense-tp-size 1 --prefill-max-requests 4 \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port ${SERVER_PORT} \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
  --disaggregation-ib-device "$IB_DEVICES" \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 8 \
  --moe-dp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config "$DEEPEP_CONFIG" \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --ep-num-redundant-experts 0 \
  --ep-dispatch-algorithm static \
  --disable-shared-experts-fusion \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --attention-context-parallel-size 8 \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --dcp-size 1 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --context-length 163840 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 32 \
  --scheduler-recv-interval 1 \
  --mem-fraction-static 0.80 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --disable-radix-cache \
  --disable-custom-all-reduce 2>&1 | tee "$LOG_DIR/prefill-rd-rank${NODE_RANK}.log"
