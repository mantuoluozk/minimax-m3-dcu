# MiniMax-M3 基线与 P/D 优化交付记录

2026-11-19。模型为 MiniMax-M3 Channel FP8 W8A8，P/D 主 KV 均为 FP8 E4M3、page64。D 的 EAGLE3 draft 使用 BF16 KV。下文将基础镜像内预装版本、实际实验基线和交付运行时分别列明。

## 基础镜像与源码

| 项目 | 版本 |
|---|---|
| 验证基础镜像 | 42.228.13.241:666/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20260920-0101 |
| 基础镜像 ID | sha256:4189a3af160467901207e0ee80b8e911ee24e32fddeee06dd9977d1130bf08a9 |
| 基础镜像预装包 | SGLang 0.5.18+dtk2604.torch2110.2609172039.gfc7ebb；PyTorch 2.11.0+dtk2604.2609171006.gec573b；Triton 3.6.0+dtk2604.torch2110.2609171456.g778eeb。由基础镜像内的包元数据读出。 |
| 海光源码线 | [release/20260825_v0.5.18](https://github.com/HYGON-AI/sglang-das/tree/release/20260825_v0.5.18) |
| 实验锁定源码 | [2c841655fe7852cde42eaaff2c1d414cf4038df1](https://github.com/HYGON-AI/sglang-das/commit/2c841655fe7852cde42eaaff2c1d414cf4038df1)，实验时由 P 的源码路径加载 |
| M3.1 参考 | 研发源码 a47dd532b371da0ffa013e299a843534ba3338e1、LightOp f162c8876e28694ef54b322722d9f67437b86814 |
| 12/37 已测镜像 | 上述基础镜像加 -sglang-212939e 后缀。现场脚本后来开启 D Graph full 和 Triton FP8 主 Attention，并关闭 multi-Q verify。仓库脚本已整理这套现场配置。 |

实验基线已经使用用户验证过的 P/D Triton Attention、FP8 KV、Mooncake、DeepEP/DeepGEMM 和 EAGLE3。11 月初先恢复这一配方，统一两端运行时，在 D 补 HCU 通信预热，P 的 prefill chunk 和 token 上限收至 32768，最多同批预填充四个请求。基础镜像的原装 SGLang 未作为端到端 A/B 控制组。可复现的控制组为迁移前源码快照加现场脚本，详细来源见 [源码清单](../deploy/source-manifest.json)。

早期交付提交 212939e 的 D 脚本默认关闭 Decode Graph 和 Triton FP8 主 Attention，并开启 multi-Q verify。本版脚本按 12/37 实测结果调整了这三项，以及 P 配置文件的相对路径和日志落盘。镜像中的 SGLang 源码和 D overlay 沿用已测快照。 [原交付提交](https://github.com/mantuoluozk/minimax-m3-dcu/tree/212939e6ccf56719b73565d910378f79338f8278)

## P 端改动与结果

| 改动 | 交付状态与对照 |
|---|---|
| QK/index Norm + RoPE | P 使用 gfx938 HIP 实现。256/2048 行单算子图回放分别约 2.27× / 3.39×，早期 7340 输入短输出 C1 的 TTFT 中位数 7.253 → 7.090 s。 |
| CP-v2 普通层兼容 | 普通 Attention 层 gather 全量 Q/K/V 后重分片，稀疏层继续 query-sharded CP。解决普通层分片和 metadata 不符，普通层有重复计算。 |
| CP FP8 packet | 主 K/V 按 checkpoint scale 打包，索引 K 仍用 BF16。P 内 CP payload 2304 → 1280 bytes/token。同布局下 C1 TTFT 1.2735 → 1.1399 s，C16 输出吞吐 74.08 → 75.39 tok/s。 |
| DeepGEMM E16 表 | 现拓扑 EP8、零冗余，适配 contiguous 规则 E17 → E16 并验证命中。C16 75.39 → 75.69 tok/s，处于小样本波动范围。 |
| OAI 激活/量化融合、raw gather/scatter | 已完成回迁候选，默认关闭。OAI 在 1952/128 C16 的吞吐 −2.46%，raw gather/scatter 在 7342/128 C16 的吞吐 −1.25%。 |

在 12/13 上以 7342 输入 / 128 输出复跑迁移前配方，保持 EAGLE3，整体迁移后 C1 TTFT 7.254 → 1.140 s，C16 TTFT 95.208 → 11.921 s，C16 窗口输出 16.55 → 75.39 tok/s。C1 TPOT 85.84 → 99.22 ms，C16 TPOT 27.71 → 72.68 ms。这组比较包含布局、调度和算子等多项变化。 [原始汇总](../benchmarks/migration-vs-pre-m31-summary.json)

## D 端改动与结果

| 改动 | 交付状态与对照 |
|---|---|
| Decode Graph full | 恢复海光 M3 已有能力。保留 EAGLE3，关闭现场曾触发 VMFault 的 multi-Q verify，回退普通 scorer。捕获并回放 target verify、draft decode、draft extend。 |
| Triton FP8 主 Attention | 启用 SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=1。主模型稀疏 Attention 使用 FP8 计算路径，索引 K 保持模型精度，draft KV 保持 BF16。 |
| DeepEP dispatch 上限 | 交付默认 128。64 版通过当前负载，7342/128 C16 111.83 → 112.04 tok/s（+0.19%）。 |
| 共享专家接收窗口重叠 | 适配候选有真实路径命中，默认关闭。12/13 的 7342/128 C16 吞吐 +0.17%。 |

同一 12/37 镜像、相同输入下，Graph 和 FP8 主 Attention 的增量依次实测。7342/128 的 C1 为两轮均值，C16 用形状预热后的正式窗口。4096/1024 每配置一条。

| 场景 | Graph off | Graph full | Graph full + FP8 主 Attention |
|---|---:|---:|---:|
| 7342/128 C1 TPOT | 100.80 ms | 41.96 ms | 22.05 ms |
| 7342/128 C1 输出吞吐 | 9.11 tok/s | 19.24 tok/s | 31.36 tok/s |
| 7342/128 C16 输出吞吐 | 67.98 tok/s | 97.26 tok/s | 111.83 tok/s |
| 7342/128 C16 整批耗时 | 30.13 s | 21.06 s | 18.31 s |
| 4096/1024 C1 TPOT | 61.28 ms | 27.39 ms | 15.27 ms |
| 4096/1024 C1 输出吞吐 | 16.13 tok/s | 35.64 tok/s | 62.67 tok/s |

7342/128 每配置 35 条请求通过检索码、token 数及完整输出检查。C16 客户端 TPOT 会受 SSE 分批到达影响，表中采用窗口吞吐和整批耗时。 [现场对照汇总](../benchmarks/site-12-37)

## 当前交付配置

P 采用 QK HIP、CP-v2、FP8 packet、DeepEP normal 和 E16 配置。D 采用 EAGLE3、普通 verify scorer、DeepEP auto、dispatch 128、Graph full 和 Triton FP8 主 Attention。部署入口和 JSON 覆盖方式见 [README](../README.md)。

在独立构建机上用上述基础镜像完成 Docker 构建和 gfx938 扩展编译。容器内检查了脚本语法、两份 JSON 和 P/D 的实际导入路径。临时遮蔽预装 SGLang 及其发行包元数据后重复导入检查通过。此轮验证范围为 CPU 构建和容器静态检查。远端实测数据来自前述 12/13 和 12/37。

## 海光分支更新核对

2026-11-19，通过 Git 远端引用确认该分支头为 0a1a19a74b466a1a3e96fdd52ef6031d305aeeac。分支头提交增加 gfx946 平台识别、JIT 下的 FP8 类型选择及 W4A8 调优配置架构识别。这套交付面向 BW1100 gfx938。 [分支头提交](https://github.com/HYGON-AI/sglang-das/commit/0a1a19a74b466a1a3e96fdd52ef6031d305aeeac)

GitHub 提交比较确认 2c841655f 是该分支头的祖先，分支头领先 3 个提交，共改动 13 个文件，涉及 gfx946、对话模板管理及其测试、HCU CI。另逐一比对了 19 个重点文件，M3 模型、QK Norm/RoPE、稀疏 Prefill/Decode、FlashMLA 适配、DeepEP、DeepGEMM、EAGLE3、DP 调度、Graph、Mooncake 和参数入口保持一致。本次继续使用已验证的运行时快照。 [上游提交对照](https://github.com/HYGON-AI/sglang-das/compare/2c841655fe7852cde42eaaff2c1d414cf4038df1...0a1a19a74b466a1a3e96fdd52ef6031d305aeeac)
