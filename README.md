# MiniMax-M3 DCU SGLang

基于海光 SGLang DAS 的 MiniMax-M3 Channel FP8 W8A8 适配，面向 gfx938。

## 在基础容器里编译

选用与宿主机驱动匹配的 BW1100 SGLang 基础镜像，或带 gfx938 工具链的海光超节点基础镜像。

- BW1100，已在此基础镜像验证编译和导入
  `42.228.13.241:666/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20260920-0101`
- 超节点基础镜像示例，按现场 DTK 和依赖版本选用
  `42.228.13.241:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-scalex40-sf_b020-20260923-1040`

在已有 SGLang 容器里执行以下命令。这里以 `/public/minimax-m3` 为部署目录，可换成任意容器内可写目录。源码放在 `sglang-source/`，部署脚本和 JSON 放在它的上一级。

~~~bash
mkdir -p /public/minimax-m3
cd /public/minimax-m3
git clone --depth 1 https://github.com/mantuoluozk/minimax-m3-dcu.git sglang-source
cd sglang-source

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${ROCM_HOME:-/opt/dtk}/.hyhal/rocm_smi/lib"
~~~

在 P 容器中编译 HIP 扩展并检查导入。

~~~bash
python3 sglang-das/hcu_m3/build_native.py
PYTHONPATH="$PWD/sglang-das/python:$PWD/build" python3 -c \
  'import torch, sglang, m31_minimax_ops; print("P:", sglang.__file__, m31_minimax_ops.__file__)'
~~~

在 D 容器中准备源码并检查导入。

~~~bash
python3 sglang-das/hcu_m3/prepare_decode.py
PYTHONPATH="$PWD/runtime/decode" python3 -c \
  'import sglang; print("D:", sglang.__file__)'
~~~

按角色运行对应的导入检查。角色脚本用 `PYTHONPATH` 优先加载本仓库的 SGLang Python 源码。D 的 `prepare_decode.py` 只生成应用了 D 端适配的源码目录。P 的 QK HIP 路径需要编译 `m31_minimax_ops.so`，完成时会打印 `M31_CANDIDATE_NATIVE_BUILD_PASS`。

## 优化内容

- **P 端**采用 QK Norm/RoPE HIP、CP-v2 普通层兼容和 FP8 KV packet，包含 E16 DeepGEMM 适配记录。
- **D 端**开启 Decode Graph full 和 Triton FP8 主 Attention，使用 HCU 通信预热和 EAGLE3，关闭此前触发 VMFault 的 multi-Q verify 打分。
- 基础环境、完整 P/D 和 Router 脚本、curl 示例、DeepEP/DeepGEMM JSON 和实测对照见 [P/D 优化与部署文档](docs/optimization.md)。
