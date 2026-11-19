# MiniMax-M3 DCU SGLang

基于海光 SGLang DAS 的 MiniMax-M3 Channel FP8 W8A8 适配，面向 gfx938。

## 在基础容器里编译

选用与宿主机驱动匹配的 BW1100 SGLang 基础镜像，或带 gfx938 工具链的海光超节点基础镜像。

- BW1100，已在此基础镜像验证编译和导入
  `42.228.13.241:666/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20260920-0101`
- 超节点基础镜像示例，按现场 DTK 和依赖版本选用
  `42.228.13.241:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-scalex40-sf_b020-20260923-1040`

基础容器需要 Git、Python/PyTorch、Ninja、C++ 和 HIP 编译器，以及兼容的 Triton、DeepEP、DeepGEMM、LightOp、Mooncake 和 Router。预装 SGLang 可选。以下命令在已经进入的基础容器内执行，默认源码目录为 `/opt/minimax-m3-dcu`。

~~~bash
git clone --depth 1 https://github.com/mantuoluozk/minimax-m3-dcu.git /opt/minimax-m3-dcu
cd /opt/minimax-m3-dcu

export ROCM_HOME=/opt/dtk
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ROCM_HOME/.hyhal/rocm_smi/lib"
export PYTORCH_ROCM_ARCH=gfx938
export MAX_JOBS=2

python3 sglang-das/hcu_m3/prepare_decode.py
python3 sglang-das/hcu_m3/build_native.py

PYTHONPATH="$PWD/sglang-das/python:$PWD/build" python3 -c \
  'import torch, sglang, m31_minimax_ops; print("P:", sglang.__file__, m31_minimax_ops.__file__)'
PYTHONPATH="$PWD/runtime/decode" python3 -c \
  'import sglang; print("D:", sglang.__file__)'
~~~

编译成功会打印 `M31_CANDIDATE_NATIVE_BUILD_PASS`。P 的导入路径应为 `sglang-das/python`，D 为 `runtime/decode`。部署时文档里的角色脚本会按相同的 `PYTHONPATH` 选择运行时。

## 优化内容

- **P 端**采用 QK Norm/RoPE HIP、CP-v2 普通层兼容和 FP8 KV packet，包含 E16 DeepGEMM 适配记录。
- **D 端**保留 HCU 通信预热和 EAGLE3，现场验证配置采用 Decode Graph full 和 Triton FP8 主 Attention，关闭此前触发 VMFault 的 multi-Q verify 打分。
- 基础环境、完整 P/D 和 Router 脚本、curl 示例、DeepEP/DeepGEMM JSON 和实测对照见 [P/D 优化与部署文档](docs/optimization.md)。
