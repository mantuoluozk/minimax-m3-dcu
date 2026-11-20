# MiniMax-M3 DCU SGLang

基于海光 SGLang DAS 的 MiniMax-M3 Channel FP8 W8A8 适配，面向 gfx938。

## 在基础容器里编译

选用与宿主机驱动匹配的 BW1100 SGLang 基础镜像，或带 gfx938 工具链的海光超节点基础镜像。

- BW1100，已在此基础镜像验证编译和导入
  `42.228.13.241:666/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20260920-0101`
- 超节点基础镜像示例，按现场 DTK 和依赖版本选用
  `42.228.13.241:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-scalex40-20261016-0200`

两台机器使用同一基础镜像。用已有的 `launch.sh` 进入容器，在部署目录执行下面这一组命令，每个环境同时准备 P 和 D。源码放在 `sglang-source/`，部署脚本和 JSON 放在它的上一级。

~~~bash
git clone --depth 1 https://github.com/mantuoluozk/minimax-m3-dcu.git sglang-source
cd sglang-source
python3 sglang-das/hcu_m3/build_native.py
python3 sglang-das/hcu_m3/prepare_decode.py
~~~

编译完成会打印 `M31_CANDIDATE_NATIVE_BUILD_PASS`。P 使用生成的 `m31_minimax_ops.so`，D 使用生成的 `runtime/decode/`。如果两个容器看到同一份共享源码目录，在该目录执行一次即可。切换 P/D 时直接使用对应脚本。

## 优化内容

- **P 端**增加 gfx938 的 QK/index Norm + RoPE HIP 路径，在海光原有 CP-v2 上补普通 Attention 层兼容与 FP8 KV packet 交换，并适配 EP8 零冗余专家布局的 DeepGEMM E16 调优规则。
- **D 端**在交付配方中开启原分支已有的 Decode Graph full 与 Triton FP8 主 Attention，保留 EAGLE3，补 HCU 通信组预热，并因 37 节点的历史 VMFault 关闭 multi-Q verify。
- 基础环境、完整 P/D 和 Router 脚本、curl 示例、DeepEP/DeepGEMM JSON 和实测对照见 [P/D 优化与部署文档](docs/optimization.md)。
