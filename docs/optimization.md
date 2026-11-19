# MiniMax-M3 基线与 P/D 优化交付记录

2026-11-19。模型为 MiniMax-M3 Channel FP8 W8A8，P/D 主 KV 均为 FP8 E4M3、page64。D 的 EAGLE3 draft 使用 BF16 KV。下文将基础镜像内预装版本、实际实验基线和交付运行时分别列明。

## 1. 环境和版本

| 项目 | 版本 |
|---|---|
| 验证基础镜像 | 42.228.13.241:666/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20260920-0101 |
| 基础镜像 ID | sha256:4189a3af160467901207e0ee80b8e911ee24e32fddeee06dd9977d1130bf08a9 |
| 基础镜像预装包 | SGLang 0.5.18+dtk2604.torch2110.2609172039.gfc7ebb；PyTorch 2.11.0+dtk2604.2609171006.gec573b；Triton 3.6.0+dtk2604.torch2110.2609171456.g778eeb。由基础镜像内的包元数据读出。 |
| 海光源码线 | [release/20260825_v0.5.18](https://github.com/HYGON-AI/sglang-das/tree/release/20260825_v0.5.18) |
| 实验锁定源码 | [2c841655fe7852cde42eaaff2c1d414cf4038df1](https://github.com/HYGON-AI/sglang-das/commit/2c841655fe7852cde42eaaff2c1d414cf4038df1)，实验时由 P 的源码路径加载 |
| M3.1 参考 | 研发源码 a47dd532b371da0ffa013e299a843534ba3338e1、LightOp f162c8876e28694ef54b322722d9f67437b86814 |
| 12/37 已测镜像 | 上述基础镜像加 -sglang-212939e 后缀。D 端开启 Decode Graph full 和 Triton FP8 主 Attention，并关闭 multi-Q verify。下方脚本整理了这套现场配置。 |

实验基线已经使用用户验证过的 P/D Triton Attention、FP8 KV、Mooncake、DeepEP/DeepGEMM 和 EAGLE3。11 月初先恢复这一配方，统一两端运行时，在 D 补 HCU 通信预热，P 的 prefill chunk 和 token 上限收至 32768，最多同批预填充四个请求。基础镜像的原装 SGLang 未作为端到端 A/B 控制组。可复现的控制组为迁移前源码快照加现场脚本，详细来源见 [源码清单](../sglang-das/hcu_m3/source-manifest.json)。

部署拓扑为两台各 8×BW1100（gfx938），P 为 TP8/CP8/EP8，D 为 TP8/DP8/EP8。两端用同一基础镜像，按 [README](../README.md) 同时准备 P 和 D 的运行时。交换角色时，按目标节点调整脚本中的网卡和 Router 地址，并在运行 D 的节点挂载 EAGLE3 draft。容器沿用现场的 launch.sh，使用 host 网络并挂载宿主机驱动目录、DCU/RDMA 设备和主模型。以 /public/minimax-m3 为例，将本页的脚本和 JSON 保存到该目录，源码克隆到同目录的 sglang-source/。源码另放时，修改脚本顶部的 PROJECT_ROOT。P 从脚本同目录读取两份 JSON，日志输出到各脚本同目录的 logs/。

## 2. 优化项和实测结果

- **P 端**使用 gfx938 QK/index Norm + RoPE HIP、CP-v2 普通 Attention 兼容和 FP8 KV packet，按 EP8 零冗余布局适配 DeepGEMM E16。
- **D 端**开启 Decode Graph full 和 Triton FP8 主 Attention，使用 EAGLE3 和普通 verify scorer。

### 4K 输入 / 1K 输出 · 单并发

优化后结果来自 12/37 两台 BW1100，4096 个随机输入 token、1024 个输出 token，完成一条请求。整套 P/D 优化前缺少这个口径的完整实测，先留空。

| 指标 | P/D 优化前 | P/D 优化后 |
|---|---:|---:|
| 首 token 延迟 TTFT | | 706.95 ms |
| 每输出 token 延迟 TPOT | | 15.27 ms |
| 输出吞吐 | | 62.67 tok/s |
| 端到端耗时 | | 16.33 s |

## 3. P.sh

保存为 P.sh，填本机控制网卡、RDMA 列表和主模型路径。P 的端口默认 40000，Mooncake bootstrap 默认 8998。

~~~bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/sglang-source"
MODEL_PATH=/model/MiniMax-M3-Channel-FP8-w8a8
NETWORK_IFACE=p14p2
IB_DEVICES=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10
SERVER_PORT=40000
BOOTSTRAP_PORT=8998
DEEPEP_CONFIG="$SCRIPT_DIR/deepep-prefill.json"
DEEPGEMM_CONFIG="$SCRIPT_DIR/deepgemm-e16.json"

export PYTHONPATH="$PROJECT_ROOT/sglang-das/python:$PROJECT_ROOT/build"

export NCCL_SOCKET_IFNAME="$NETWORK_IFACE"
export GLOO_SOCKET_IFNAME="$NETWORK_IFACE"

export SGLANG_ENABLE_CP_V2=1
export SGLANG_MINIMAX_CP_FP8_KV_EXCHANGE=1
export SGLANG_HEALTH_CHECK_TIMEOUT=120
export SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=1

export NCCL_IB_HCA="${IB_DEVICES//,/:1,}:1"
export ROCSHMEM_ALLOWED_IBV_DEVICES="$IB_DEVICES"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="$IB_DEVICES"
export MC_ALLOWED_IBV_DEVICES="$IB_DEVICES"

export SGLANG_HCU_DEEPGEMM_CONTIG_TUNING_CONFIG="$DEEPGEMM_CONFIG"

unset SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER SGLANG_USE_FUSED_TOPK_SOFTMAX
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_KV_LAYOUT_HCU_FA=false
export SGLANG_REQUIRE_KV_CACHE_SCALES=1
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

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/prefill-$(date +%Y%m%d-%H%M%S)-$$.log"
printf "Log file: %s\n" "$LOG_FILE"

python3 -u -m sglang.launch_server \
  --dist-init-addr "${DIST_INIT_ADDR:-127.0.0.1:26000}" \
  --enable-dp-attention --disable-overlap-schedule \
  --moe-dense-tp-size 1 --prefill-max-requests 4 \
  --model-path "$MODEL_PATH" \
  --served-model-name MiniMax-M3 \
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
~~~

改好脚本顶部的模型路径和网卡后执行 `bash P.sh`。

## 4. D.sh

保存为 D.sh，填本机控制网卡、RDMA 列表、主模型和 draft 路径。D 的端口默认 40001。首次 Graph 捕获完成后服务才会就绪。

~~~bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/sglang-source"
MODEL_PATH=/model/MiniMax-M3-Channel-FP8-w8a8
NETWORK_IFACE=p14p1
IB_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
SERVER_PORT=40001
DRAFT_PATH=/model/MiniMax-M3-EAGLE3

export PYTHONPATH="$PROJECT_ROOT/runtime/decode"

export NCCL_SOCKET_IFNAME="$NETWORK_IFACE"
export GLOO_SOCKET_IFNAME="$NETWORK_IFACE"

export SGLANG_HEALTH_CHECK_TIMEOUT=120
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_KV_LAYOUT_HCU_FA=false

export NCCL_IB_HCA="${IB_DEVICES//,/:1,}:1"
export ROCSHMEM_ALLOWED_IBV_DEVICES="$IB_DEVICES"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="$IB_DEVICES"
export MC_ALLOWED_IBV_DEVICES="$IB_DEVICES"

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600

export SGLANG_REQUIRE_KV_CACHE_SCALES=1
export SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1

export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
unset SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_MINIMAX_ROUTER_GEMV_BLOCK_K=1024
export SGLANG_MINIMAX_ROUTER_GEMV_NUM_WARPS=1

export SGLANG_OPT_USE_EAGLE3_LM_HEAD_TOP1=1
# Use the standard verify scorer after the node37 multi-Q VMFault.
unset SGLANG_OPT_USE_MINIMAX_MULTI_Q_VERIFY_SCORE
export SGLANG_MINIMAX_MTP_NUM_TOPK_CHUNKS=4

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/decode-$(date +%Y%m%d-%H%M%S)-$$.log"
printf "Log file: %s\n" "$LOG_FILE"

python3 -u -m sglang.launch_server \
  --dist-init-addr "${DIST_INIT_ADDR:-127.0.0.1:26320}" \
  --model-path "$MODEL_PATH" \
  --served-model-name MiniMax-M3 \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$IB_DEVICES" \
  --tp-size 8 \
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
  --ep-dispatch-algorithm static \
  --disable-shared-experts-fusion \
  --attention-backend triton \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
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
  --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 1 2 3 4 8 16 \
  --cuda-graph-backend-prefill disabled \
  --context-length 163840 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --max-running-requests 128 \
  --mem-fraction-static 0.80 \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --pre-warm-nccl \
  --disable-radix-cache \
  --disable-custom-all-reduce 2>&1 | tee "$LOG_FILE"
~~~

改好脚本顶部的模型路径和网卡后执行 `bash D.sh`。

## 5. router.sh

保存为 router.sh。P 和 D 的 /v1/models 都可访问后，填实际服务地址并启动 Router。默认端口 40010。

~~~bash
#!/usr/bin/env bash
set -euo pipefail
: "${PREFILL_URL:?Set PREFILL_URL}"
: "${DECODE_URL:?Set DECODE_URL}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/router-$(date +%Y%m%d-%H%M%S)-$$.log"
python3 -u -m sglang_router.launch_router \
  --pd-disaggregation --prefill "$PREFILL_URL" --decode "$DECODE_URL" \
  --host 0.0.0.0 --policy round_robin --port "${ROUTER_PORT:-40010}" \
  --prometheus-port "${PROMETHEUS_PORT:-3035}" \
  --health-check-endpoint /v1/models --health-check-interval-secs 60 \
  --health-check-timeout-secs 3600 --request-timeout-secs 36000 2>&1 | tee "$LOG_FILE"
~~~

启动示例中的设备名和地址按现场替换。

~~~bash
PREFILL_URL='http://10.20.100.12:40000' DECODE_URL='http://10.20.100.37:40001' bash router.sh
~~~

## 6. curl 冒烟验证

~~~bash
ROUTER=http://10.20.100.12:40010
curl --noproxy '*' -fsS "$ROUTER/workers"
curl --noproxy '*' -fsS "$ROUTER/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3","messages":[{"role":"user","content":"验证码是7359。请只输出验证码。"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"thinking_mode":"disabled"}}'
~~~

确认两个 worker 健康且响应包含 7359。正式性能对照使用长输出请求，通过 D 日志核对 Triton FP8 主 Attention 和 Graph 回放。

## 7. P 端参数文件

将以下两个完整 JSON 保存到 P.sh 所在目录。更换配置时，修改 P.sh 顶部的 DEEPEP_CONFIG 和 DEEPGEMM_CONFIG。第一份控制 DeepEP normal dispatch 和 combine，第二份按 E/N/K 及 M 区间匹配 DeepGEMM contiguous GEMM。

### deepep-prefill.json

~~~json
{
  "normal_dispatch": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 16,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  },
  "normal_combine": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 2,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  }
}
~~~

### deepgemm-e16.json

~~~json
{
  "version": 1,
  "topology": "TP8/CP8/EP8/R0; candidate E16 adaptation",
  "coverage": {
    "actual_batch_sizes": [
      2,
      4,
      8
    ],
    "observed_M_min": 7409,
    "observed_M_max": 112236,
    "representative_quantiles": [
      0.1,
      0.5,
      0.9
    ]
  },
  "rules": [
    {
      "E": 16,
      "N": 6144,
      "K": 6144,
      "min_M": 7409,
      "max_M": 112236,
      "config": {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "N_LOOP_NUM": 4
      }
    },
    {
      "E": 16,
      "N": 6144,
      "K": 3072,
      "min_M": 7409,
      "max_M": 112236,
      "config": {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "N_LOOP_NUM": 4
      }
    }
  ],
  "adaptation": {
    "source_sha256": "c50d9bb7f80669074bb5a2b6556a2727d5b74e115cecf5ec2a8a53322cf064d5",
    "changed": "E17 to E16 only; original M bounds and block parameters retained",
    "status": "candidate, not retuned optimum"
  }
}
~~~

## 海光分支更新核对

2026-11-19，通过 Git 远端引用确认该分支头为 0a1a19a74b466a1a3e96fdd52ef6031d305aeeac。分支头提交增加 gfx946 平台识别、JIT 下的 FP8 类型选择及 W4A8 调优配置架构识别。这套交付面向 BW1100 gfx938。 [分支头提交](https://github.com/HYGON-AI/sglang-das/commit/0a1a19a74b466a1a3e96fdd52ef6031d305aeeac)

GitHub 提交比较确认 2c841655f 是该分支头的祖先，分支头领先 3 个提交，共改动 13 个文件，涉及 gfx946、对话模板管理及其测试、HCU CI。另逐一比对了 19 个重点文件，M3 模型、QK Norm/RoPE、稀疏 Prefill/Decode、FlashMLA 适配、DeepEP、DeepGEMM、EAGLE3、DP 调度、Graph、Mooncake 和参数入口保持一致。本次继续使用已验证的运行时快照。 [上游提交对照](https://github.com/HYGON-AI/sglang-das/compare/2c841655fe7852cde42eaaff2c1d414cf4038df1...0a1a19a74b466a1a3e96fdd52ef6031d305aeeac)
