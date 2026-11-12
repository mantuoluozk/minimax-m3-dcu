#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NODE_RANK=0
SERVER_PORT=${SERVER_PORT:-40001}
export SGLANG_HEALTH_CHECK_TIMEOUT=120
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_M31_DECODE_SHARED_RECV=0
: "${DRAFT_PATH:?Set DRAFT_PATH inside the container}"
export PYTHONPATH="$PROJECT_ROOT/runtime/decode"
export SGLANG_KV_LAYOUT_HCU_FA=false



mkdir -p "$LOG_DIR"

export ROCM_HOME=/opt/dtk
export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk/hip



export NCCL_IB_HCA="${NCCL_IB_HCA:-${IB_DEVICES//,/:1,}:1}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${IB_DEVICES}"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="${IB_DEVICES}"
export MC_ALLOWED_IBV_DEVICES="${IB_DEVICES}"

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600

export SGLANG_REQUIRE_KV_CACHE_SCALES=1
export SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1

export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938=0
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_INDEXER=0

export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
unset SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_MINIMAX_ROUTER_GEMV_BLOCK_K=1024
export SGLANG_MINIMAX_ROUTER_GEMV_NUM_WARPS=1

export SGLANG_OPT_USE_EAGLE3_LM_HEAD_TOP1=1
export SGLANG_OPT_USE_MINIMAX_MULTI_Q_VERIFY_SCORE=1
export SGLANG_MINIMAX_MTP_NUM_TOPK_CHUNKS=4




export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_DECODE=0
export SGLANG_OPT_USE_BF16_ROUTER_GEMM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
python3 -m sglang.launch_server --dist-init-addr "${DIST_INIT_ADDR:-127.0.0.1:26320}" \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port ${SERVER_PORT} \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$IB_DEVICES" \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 8 \
  --load-balance-method round_robin \
  --enable-dp-attention \
  --enable-dp-attention-local-control-broadcast \
  --enable-dp-lm-head \
  --ep-size 8 \
  --moe-dp-size 1 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --deepep-dispatcher-output-dtype fp8 \
  --moe-runner-backend deep_gemm \
  --ep-num-redundant-experts 0 \
  --ep-dispatch-algorithm static \
  --disable-shared-experts-fusion \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --attention-context-parallel-size 1 \
  --dcp-size 1 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$DRAFT_PATH" \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-kv-cache-dtype bf16 \
  --speculative-draft-attention-backend triton \
  --speculative-attention-mode prefill \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-draft-window-size 4096 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 2 3 4 8 16 \
  --context-length 163840 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --max-running-requests 128 \
  --scheduler-recv-interval 1 \
  --mem-fraction-static 0.80 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --pre-warm-nccl \
  --disable-radix-cache \
  --disable-custom-all-reduce 2>&1 | tee "$LOG_DIR/decode-rd-rank${NODE_RANK}.log"
