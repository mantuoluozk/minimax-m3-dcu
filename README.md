# MiniMax-M3 DCU

MiniMax-M3 Channel FP8 W8A8 的海光 DCU 适配。验证拓扑为两台 8×BW1100，P 端 CP8，D 端 DP8/EP8，Mooncake 传输与 EAGLE3。

## 从已有镜像构建

先选一个与目标 DCU 和宿主机驱动匹配的 Linux AMD64 推理镜像，再把本仓库的 SGLang 源码和原生扩展构建进新镜像。基础镜像可通过 `BASE_IMAGE` 自选，需要提供兼容的 Python、PyTorch、DTK/HIP、DeepEP、DeepGEMM、LightOp、Mooncake 和 SGLang Router。P 端构建还需要 HIP 编译器、C++ 编译器和 Ninja。当前 Dockerfile 按 `/opt/dtk` 和 `gfx938` 构建扩展。

```bash
git clone https://github.com/mantuoluozk/minimax-m3-dcu.git
cd minimax-m3-dcu

BASE_IMAGE='<你的海光推理基础镜像:标签>'
IMAGE='minimax-m3-dcu:custom'
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" .
```

Dockerfile 把仓库源码复制进镜像，生成 D 端运行时，并用基础镜像内的工具链编译 P 端原生扩展。启动时，`prefill` 和 `decode` 入口分别设置自己的 `PYTHONPATH`，使服务优先加载仓库里的 SGLang。也就是说，服务运行时用的是适配版本，基础镜像提供依赖。

构建后可核对 Python 实际解析到的包路径：

```bash
docker run --rm --entrypoint python3 \
  -e PYTHONPATH=/opt/minimax-m3-dcu/sglang-das/python:/opt/minimax-m3-dcu/build \
  "$IMAGE" -c 'import importlib.util; print(importlib.util.find_spec("sglang").origin)'

docker run --rm --entrypoint python3 \
  -e PYTHONPATH=/opt/minimax-m3-dcu/runtime/decode \
  "$IMAGE" -c 'import importlib.util; print(importlib.util.find_spec("sglang").origin)'
```

P 端应解析到 `/opt/minimax-m3-dcu/sglang-das/python/sglang/__init__.py`，D 端应解析到 `/opt/minimax-m3-dcu/runtime/decode/sglang/__init__.py`。只为 D 构建时可以传 `--build-arg BUILD_NATIVE=0` 跳过 P 端扩展编译。更换基础镜像后，在目标环境验证 P/D 启动和长请求。

## 源码布局

- `sglang-das/` 保存完整 SGLang DAS 源码，基于公开提交 `2c841655fe7852cde42eaaff2c1d414cf4038df1`。其中 `python/sglang` 取自已测 P 运行时，含 QK HIP、CP-v2 普通 Attention 兼容、FP8 packet 和 DeepGEMM E16 适配。
- `deploy/decode-overlay/` 在构建时还原已测 D 运行时。D 保留 EAGLE3 和启动通信预热。原重叠候选补丁位于 `experiments/decode-recv-overlap/`，仅供复核。
- `native/` 保存 LightOp 候选源码和 bindings。构建时使用基础镜像自带的 Torch、DTK、HIP 编译 gfx938 扩展。各依赖由基础镜像提供。
- `benchmarks/` 保存性能汇总，`deploy/source-manifest.json` 记录源快照和差异。模型、draft 权重、原容器和逐请求日志独立保存。

`experiments/prefill-cp/` 保留 CP、QK 及 raw scatter 的阶段补丁，`experiments/oai-triton/` 保留按当前 Triton 量化契约适配的融合候选。候选文件对应各自实验基线，默认镜像使用已测 QK、CP-v2 和 FP8 packet。

## 启动容器和服务

在 P、D 各自的宿主机上，从仓库根目录执行 `docker.sh`。脚本沿用服务器 `launch_docker.sh` 的 host 网络、host IPC、DCU 设备和 `/opt/hyhal`、`/data/xsf`、`/model` 挂载，并把当前仓库只读挂载到容器内 `/workspace/minimax-m3-dcu`。脚本沿用服务器的特权容器配置，请在具备相应权限的节点运行。

```bash
# P 宿主机
DOCKER_IMAGE=minimax-m3-dcu:custom CONTAINER_NAME=minimax-m3-p bash docker.sh

# D 宿主机
DOCKER_IMAGE=minimax-m3-dcu:custom CONTAINER_NAME=minimax-m3-d bash docker.sh
```

如果 P、D 分别构建，给 `DOCKER_IMAGE` 填各自的标签。默认命令会进入容器 shell，`-v` 启动容器后只打印进入命令。可以在创建容器时用 `DOCKER_ENV_FILE` 指定 Docker 环境变量文件，或者进入容器后再设置网卡、RDMA 设备和模型路径。

```bash
# P 容器内
cd /workspace/minimax-m3-dcu
export NETWORK_IFACE='<本机网卡>'
export IB_DEVICES='<逗号分隔的本机 RDMA 设备>'
export MODEL_PATH=/model/MiniMax-M3-Channel-FP8-w8a8
bash P.sh

# D 容器内
cd /workspace/minimax-m3-dcu
export NETWORK_IFACE='<本机网卡>'
export IB_DEVICES='<逗号分隔的本机 RDMA 设备>'
export MODEL_PATH=/model/MiniMax-M3-Channel-FP8-w8a8
export DRAFT_PATH=/model/MiniMax-M3-EAGLE3
bash D.sh

# 等 P、D 健康后，在 P 容器的另一个 shell 中执行
cd /workspace/minimax-m3-dcu
PREFILL_URL="http://<P_IP>:40000" DECODE_URL="http://<D_IP>:40001" bash router.sh
```

`P.sh` 和 `D.sh` 使用镜像内 `/opt/minimax-m3-dcu` 的运行时及扩展。调整启动参数可直接编辑宿主机仓库中的角色脚本，修改 SGLang 源码或扩展后重新构建镜像。也可以直接使用镜像入口参数 `prefill`、`decode` 和 `router`。

默认端口 P 40000、D 40001、Router 40010，P bootstrap 8998。以两端 `/health`、Router `/workers` 和通过 Router 的长提示词请求确认部署。E16 JSON 是命中验证用的试验表，可用 `DEEPGEMM_CONFIG` 指定其他已验证规则。

## 已归档的镜像示例

2026-11-12 的交付归档包含 `minimax-m3-dcu:prefill-20261112` 和 `minimax-m3-dcu:decode-20261112`。当时 P、D 各自沿用原测试基础镜像，标签尾部分别为 `dtk2604-py3.10-20260920-0101` 和 `dtk26041-py3.10-20261019-0103`。两套产物通过无 DCU 设备挂载的入口、Torch 和 SGLang 导入检查，P 原生扩展编译和导入通过。

自建镜像可用 `docker save -o minimax-m3-dcu.tar "$IMAGE"` 导出，到目标机器用 `docker load -i minimax-m3-dcu.tar` 导入。

## 性能记录

历史实测使用 7342 输入、128 输出，EAGLE3 固定。迁移前复跑对比 CP FP8 配方，单并发平均 TTFT 7.2544→1.1399 秒，16 并发整批输出 16.5506→75.3942 token/s。客户端 TPOT 同时上升。C16 正式计时每配置一轮。数据来自归档的原运行时，镜像打包后需在目标环境复验。
