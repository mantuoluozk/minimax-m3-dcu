# MiniMax-M3 DCU

MiniMax-M3 Channel FP8 W8A8 的海光 DCU 适配。验证拓扑为两台 8×BW1100，P 端 CP8，D 端 DP8/EP8，Mooncake 传输与 EAGLE3。

- `sglang-das/` 保存完整 SGLang DAS 源码，基于公开提交 `2c841655fe7852cde42eaaff2c1d414cf4038df1`。其中 `python/sglang` 取自已测 P 运行时，含 QK HIP、CP-v2 普通 Attention 兼容、FP8 packet 和 DeepGEMM E16 适配。
- `deploy/decode-overlay/` 在构建时还原已测 D 运行时。D 保留 EAGLE3 和启动通信预热。原重叠候选补丁位于 `experiments/decode-recv-overlap/`，仅供复核。
- `native/` 保存 LightOp 候选源码和 bindings。构建时使用基础镜像自带的 Torch、DTK、HIP 编译 gfx938 扩展。各依赖由基础镜像提供。
- `benchmarks/` 保存性能汇总，`deploy/source-manifest.json` 记录源快照和差异。模型、draft 权重、原容器和逐请求日志独立保存。

`experiments/prefill-cp/` 保留 CP、QK 及 raw scatter 的阶段补丁，`experiments/oai-triton/` 保留按当前 Triton 量化契约适配的融合候选。候选文件对应各自实验基线，默认镜像使用已测 QK、CP-v2 和 FP8 packet。

## 构建镜像

在能够访问海光基础镜像的 Linux AMD64 Docker 构建机上运行。原实测 P、D 的基础镜像不同，使用同一 Dockerfile 时按角色传入对应镜像。P 基础镜像标签尾部为 `dtk2604-py3.10-20260920-0101`，D 为 `dtk26041-py3.10-20261019-0103`。两个角色使用同一基础镜像仍需单独验证。

```bash
docker build --build-arg BASE_IMAGE="$P_BASE_IMAGE" -t minimax-m3-dcu:prefill-20261112 .
docker build --build-arg BASE_IMAGE="$D_BASE_IMAGE" --build-arg BUILD_NATIVE=0 \
  -t minimax-m3-dcu:decode-20261112 .
```

需要镜像文件时可用 `docker save -o minimax-m3-dcu.tar minimax-m3-dcu:prefill-20261112 minimax-m3-dcu:decode-20261112`，目标机器用 `docker load -i minimax-m3-dcu.tar` 导入。

2026-11-12 打包时，两套镜像均通过无 DCU 设备挂载的启动入口、Torch 和 SGLang 导入检查，P 原生扩展编译和导入通过。镜像保留归档运行时的原始文件，Git 中两处 Mistral 类引用仅调整了换行，Python 语法树一致。

## 启动

P、D 分别在对应宿主机启动，启用 host 网络和 host IPC，挂载站点需要的 DCU/RDMA 设备、驱动和只读模型目录。下面命令假定站点设备名与示例一致，RDMA 设备列表及管理网卡从环境变量传入。

```bash
# P 节点
docker run --rm --network host --ipc host --group-add video \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --device=/dev/infiniband \
  -v /opt/hyhal:/opt/hyhal:ro -v /model:/model:ro \
  -e NETWORK_IFACE -e IB_DEVICES \
  -e MODEL_PATH=/model/MiniMax-M3-Channel-FP8-w8a8 minimax-m3-dcu:prefill-20261112 prefill

# D 节点，挂载与主模型匹配的 EAGLE3 draft
docker run --rm --network host --ipc host --group-add video \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --device=/dev/infiniband \
  -v /opt/hyhal:/opt/hyhal:ro -v /model:/model:ro \
  -e NETWORK_IFACE -e IB_DEVICES \
  -e MODEL_PATH=/model/MiniMax-M3-Channel-FP8-w8a8 \
  -e DRAFT_PATH=/model/MiniMax-M3-EAGLE3 minimax-m3-dcu:decode-20261112 decode

# Router，URL 指向实际 P 和 D 节点
docker run --rm --network host -e PREFILL_URL -e DECODE_URL minimax-m3-dcu:prefill-20261112 router
```

默认端口 P 40000、D 40001、Router 40010，P bootstrap 8998。以两端 `/health`、Router `/workers` 和通过 Router 的长提示词请求确认部署。E16 JSON 是命中验证用的试验表，可用 `DEEPGEMM_CONFIG` 指定其他已验证规则。

## 性能记录

历史实测使用 7342 输入、128 输出，EAGLE3 固定。迁移前复跑对比 CP FP8 配方，单并发平均 TTFT 7.2544→1.1399 秒，16 并发整批输出 16.5506→75.3942 token/s。客户端 TPOT 同时上升。C16 正式计时每配置一轮。数据来自归档的原运行时，镜像打包后需在目标环境复验。
