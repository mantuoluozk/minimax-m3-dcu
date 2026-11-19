# MiniMax-M3 DCU

在兼容的海光 DCU 基础镜像上构建 MiniMax-M3 Channel FP8 W8A8 的优化版 SGLang，使用同一张镜像启动两台 8×BW1100 的 Prefill 和 Decode。P 使用 CP8、FP8 KV 和 DeepEP，D 使用 DP8/EP8、EAGLE3、Triton FP8 主 Attention 和 Decode Graph full。两端使用 Mooncake RDMA 交接主 KV。

[基础镜像、默认 SGLang 与 P/D 优化记录](docs/optimization.md)

## 1. 构建镜像

基础镜像可自选。需要 Linux amd64、兼容宿主机的 DTK/HIP（路径 `/opt/dtk`）、HIP/C++ 编译器、Ninja、Python/PyTorch，以及兼容的 DeepEP、DeepGEMM、LightOp、Triton、Mooncake 和 SGLang Router 依赖。基础镜像可以已有 SGLang，也可以只提供上述运行和编译依赖。

构建时 Dockerfile 复制本仓库的 SGLang Python 源码，生成 Decode 专用运行时，并编译 gfx938 原生扩展。P/D 启动脚本分别用 `PYTHONPATH` 选中这些源码，基础镜像自带的 SGLang 安装无需处理。

~~~bash
git clone https://github.com/mantuoluozk/minimax-m3-dcu.git
cd minimax-m3-dcu

BASE_IMAGE=42.228.13.241:666/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20260920-0101
REV=$(git rev-parse --short HEAD)
IMAGE="${BASE_IMAGE}-sglang-${REV}"

docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE" .
docker run --rm --entrypoint bash "$IMAGE" -ec '
  PYTHONPATH=sglang-das/python:build python3 -c "import torch, sglang, m31_minimax_ops; print(sglang.__file__, m31_minimax_ops.__file__)"
  PYTHONPATH=runtime/decode python3 -c "import sglang; print(sglang.__file__)"
'
~~~

P 应从 `/opt/minimax-m3-dcu/sglang-das/python` 解析 SGLang，D 应从 `/opt/minimax-m3-dcu/runtime/decode` 解析。将同一镜像、同一 Git 提交的部署文件准备到 P 和 D 两台机器。跨机传输可用 docker save 和 docker load，或者推送到两端可访问的镜像仓库。权重按下节单独挂载。

## 2. 配置 P/D 和 JSON

在两台宿主机各准备一份仓库。启动时从仓库目录读取脚本和 JSON，默认把宿主机 /model 只读挂载到容器 /model。权重若在其他目录，启动容器时设置 MODEL_HOST_DIR 为权重目录的宿主机上级目录。

| 文件或变量 | 配置 |
|---|---|
| P.sh | 必填 NETWORK_IFACE、IB_DEVICES、MODEL_PATH。默认服务端口 40000、Mooncake bootstrap 8998。 |
| D.sh | 必填 NETWORK_IFACE、IB_DEVICES、MODEL_PATH、DRAFT_PATH。默认端口 40001，使用与 P 相同的主模型权重和 MiniMax-M3 EAGLE3 draft。 |
| SERVED_MODEL_NAME | 两端默认 MiniMax-M3。修改时两端填相同值。 |
| deploy/config/deepep-prefill.json | P 的 DeepEP normal 调度配置，由 P.sh 的 --deepep-config 加载。新拓扑可用容器内绝对路径 DEEPEP_CONFIG 指向对应配置。 |
| deploy/config/deepgemm-e16.json | P 的 contiguous DeepGEMM E16 试验规则，支持 DEEPGEMM_CONFIG 覆盖。E、N、K 或 M 区间不匹配时由运行时走默认路径。 |
| D 的 DeepEP | 使用 auto 模式及每 rank 128 token dispatch 上限。64 的现场对照基本持平。 |
| router.sh | 填 PREFILL_URL 和 DECODE_URL，默认 Router 端口 40010、Prometheus 3035。 |

NETWORK_IFACE 是控制面网卡名，IB_DEVICES 为本机逗号分隔的 RDMA 设备名，可在宿主机用 ip -br addr 和 ibv_devices 查询。Docker 使用 host 网络，同一宿主机的容器共享监听端口。

## 3. 启动

下面命令中的网卡、RDMA 设备列表和 IP 替换成现场值。docker.sh 进入容器交互 shell，后续启动参数在该 shell 里执行。

~~~bash
# P 宿主机，从仓库目录执行
DOCKER_IMAGE="$IMAGE" CONTAINER_NAME=minimax-m3-p bash docker.sh
~~~

~~~bash
# P 容器内
NETWORK_IFACE='<P 控制网卡>' IB_DEVICES='<P RDMA 设备列表>' \
MODEL_PATH=/model/MiniMax-M3-Channel-FP8-w8a8 bash P.sh
~~~

~~~bash
# D 宿主机，从仓库目录执行
DOCKER_IMAGE="$IMAGE" CONTAINER_NAME=minimax-m3-d bash docker.sh
~~~

~~~bash
# D 容器内
NETWORK_IFACE='<D 控制网卡>' IB_DEVICES='<D RDMA 设备列表>' \
MODEL_PATH=/model/MiniMax-M3-Channel-FP8-w8a8 \
DRAFT_PATH=/model/MiniMax-M3-EAGLE3 bash D.sh
~~~

两端 /v1/models 可访问后，在 P 宿主机另开终端进入容器启动 Router。首次 D 图捕获可能需要数分钟。

~~~bash
docker exec -it -w /workspace/minimax-m3-dcu minimax-m3-p bash
~~~

~~~bash
PREFILL_URL="http://<P_IP>:40000" DECODE_URL="http://<D_IP>:40001" bash router.sh
~~~

宿主机仓库下的 logs/ 收集各角色带时间戳的日志，仓库和配置本身只读挂载到容器。已有同名容器会被复用，新镜像或挂载参数变化时使用新的 CONTAINER_NAME。使用现有容器启动脚本也可以，挂载仓库及权重后执行对应角色脚本，并设置可写 LOG_DIR。

## 4. 验证

~~~bash
ROUTER="http://<P_IP>:40010"
curl --noproxy '*' -fsS "$ROUTER/workers"
curl --noproxy '*' -fsS "$ROUTER/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3","messages":[{"role":"user","content":"验证码是7359。请只输出验证码。"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"thinking_mode":"disabled"}}'
~~~

确认两个 worker 都健康、响应返回 7359。正式性能对照使用固定长提示词和足够的生成 token，结合 D 日志核对 triton_fp8_main_attn_gemm=True 与实际 Decode batch 的 cuda graph: True。

## 源码位置

- sglang-das/ 保存完整 DAS 源码，交付来源见 deploy/source-manifest.json。
- deploy/decode-overlay/ 保存 D 专用差异，构建时用它生成 D 的独立运行时。
- native/ 保存 HIP 扩展和绑定。experiments/ 保存候选补丁，benchmarks/ 保存对照结果。
