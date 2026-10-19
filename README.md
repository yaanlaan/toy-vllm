# Toy-vLLM: 从零构建高性能大模型推理引擎

本项目参考工业级推理系统（vLLM / SGLang）及《从零实现 vLLM》演进路径，采取**循序渐进、版本迭代（V0 -> V5）**的工程化方式，从纯 PyTorch 最小自回归原型出发，逐步演进至支持 PagedAttention、Continuous Batching、Triton 算子优化、单机多卡张量并行（TP）以及 MoE 专家并行（EP）的完整大模型推理引擎。

---

## 版本演进全景路线 (V0 ~ V5 全量达成)

- **[V0: 最小可用自回归推理原型 (已完成)](#-v0-极简自回归原型)**
  - 零显存池负担、无 KV Cache，纯 Tensor 拼接跑通端到端自回归循环。
  - 手写核心算子：RMSNorm、RoPE 旋转编码、GQA 注意力、SwiGLU 激活、多策略 Sampler。
  - 支持直接加载官方 Safetensors 权重，与 HuggingFace 官方输出保持精确数值对齐。
- **[V1: KV 缓存与分页显存管理 (PagedAttention) (已完成)](#-v1-pagedattention-与分页显存池)**
  - 实现物理块分配器 `BlockAllocator`（Free List 维护与引用计数生命周期管理，零显存泄漏）。
  - 实现逻辑到物理页映射 `BlockTable`，并在 GPU 上维护块表 Tensor 消除 CPU-to-GPU 频繁传输。
  - 实现全局统一物理显存池 `KVCachePool`，小颗粒度（`block_size=16`）将显存碎片率降至 4% 以下。
  - 重构 `GroupedQueryAttention` 支持 Prefill（批量写入）与 Decode（增量单 token 写入与历史 KV 寻址聚合）两阶段流水。
  - 贪心采样下与 V0 验证 100% 比特级一致性。
- **[V2: 连续批处理与并发调度器 (Continuous Batching) (已完成)](#-v2-连续批处理与并发调度器)**
  - 实现请求级状态机（`WAITING` -> `RUNNING` -> `FINISHED`）与 `Sequence` 生命周期抽象。
  - 实现动态迭代级调度器 `Scheduler`，支持“先保存量、再迎增量”显存安全决策。
  - 升级 `ForwardContext` 与 `GroupedQueryAttention` 支持多请求并发 Batch Decode 混合张量计算。
  - 开放 `LLMEngine.add_request()`、`LLMEngine.step()` 与 `LLMEngine.generate_batch()` 高性能并发服务接口。
  - 解决短请求等待长请求的木桶死锁，短请求完成即刻释放显存，新请求无缝插队接力。
- **[V3: 单卡算子深度融合与 CUDA Graph 优化 (已完成)](#-v3-单卡算子深度融合与-cuda-graph-优化)**
  - 基于 Triton 编写高性能 `PagedAttention` Decode 内核，由 GPU 线程块直接通过 `block_table` 寻址，实现在线重标度 Softmax 点积注意力，彻底消灭 Python 端切片索引与中间 Tensor 分配。
  - 实现 `Fused RMSNorm + Residual` 与 `Fused SwiGLU` 融合算子，将慢速 HBM 显存往返降低 60%，在片上高速 SRAM 中原位完成计算。
  - 实现 `CUDAGraphRunner` 静态图录制与回放，消除 Python 解释器在 CPU 端的发射开销（Launch Overhead）。
  - RTX 5060 Ti 实测单 Token 解码延迟（ITL）从 70.9ms 骤降至 39.7ms（削减 44.1%），端到端内核加速比达 **1.79x**！
- **[V4: 单机多卡张量并行 (Tensor Parallelism) (已完成)](#-v4-单机多卡张量并行)**
  - 封装分布式状态机 `parallel_state` 与卡间通信原语 `tensor_model_parallel_all_reduce`，支持 `MockAllReduceGroup` 线程级多卡模拟与原生分布式通信。
  - 实现 Megatron 风格正交切分线性层：`ColumnParallelLinear`（QKV / Gate / Up 投影）与 `RowParallelLinear`（Out / Down 投影），单层仅需 2 次 All-Reduce 通信。
  - 实现 `ParallelQwen2ForCausalLM`，支持加载全量 Safetensors 并按 Rank 索引自动化切片。
  - 验证多 Worker 并发前向与单卡前向保持 **$10^{-7}$ 级别的极致数值对齐**！
- **[V5: MoE 架构支持与专家并行 (Expert Parallelism) (已完成)](#-v5-moe-架构支持与专家并行)**
  - 实现 `MoERouter` Top-K 动态稀疏门控网络与局部 Softmax 归一化。
  - 支持类似 Qwen2-MoE / DeepSeek 的双重专家机制：专属性路由专家 + 基础通用通识共享专家（Shared Expert）。
  - 突破性实现**“专家找 Token”的工业级分组重排算子管线（Permute $\to$ 批量专家 GEMM $\to$ Unpermute 散布还原）**，相比传统初级循环提速 **4.1x**！
  - 封装多卡专家并行 `ExpertParallelMoEMLP` 与两次 `All-to-All` 跨卡通信流水线（Token 派发 + 结果回收），实现单卡专家显存直接砍半！

---

## 环境快速配置 (基于 uv)

本项目使用现代 Python 包管理器 `uv` 进行依赖和虚拟环境管理：

```powershell
# 1. 创建虚拟环境 (推荐 Python 3.12)
uv venv --python 3.12

# 2. 激活虚拟环境
.venv\Scripts\activate

# 3. 安装依赖 (PyTorch CUDA 12.8/13.0 支持 Blackwell RTX 50 系列及现有 GPU)
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install triton-windows transformers safetensors accelerate tokenizers sentencepiece tiktoken numpy matplotlib pytest
```

---

## 快速测试与运行指南

### 1. 运行自动化单元测试
```powershell
$env:PYTHONPATH="."

# 运行全量 20 项单元测试 (覆盖 V0~V5 全栈模块、Triton 内核与分布式并行)
python -m pytest tests/
```

### 2. 运行各版本推理与基准示例
```powershell
# [V0] 极简端到端流式推理 Demo
python examples/demo_v0.py

# [V1] Paged KV Cache 性能基准对比实测 (验证加速比与输出一致性)
python examples/demo_v1_benchmark.py

# [V2] 连续批处理多并发压测 (对比单请求串行 vs 并发动态调度吞吐)
python examples/demo_v2_continuous_batching.py

# [V3] Triton 算子融合与内核性能加速实测 (对比 PyTorch 原生 vs Triton 内核)
python examples/demo_v3_speedup.py

# [V4] 单机多卡张量并行 (TP=2) 模拟分布式推理 Demo
python examples/demo_v4_tensor_parallelism.py

# [V5] MoE 混合专家模型稀疏路由与专家并行 (EP) 实测 Demo
python examples/demo_v5_moe_inference.py
```

---

## 详细学习笔记与技术图解索引

### 阶段技术笔记文档
- **[V0 详细设计笔记与数学公式推导](docs/v0_minimal_inference.md)**
- **[V1 PagedAttention 分页显存池深度剖析](docs/v1_paged_kv_cache.md)**
- **[V2 连续批处理与迭代级动态调度器详解](docs/v2_continuous_batching.md)**
- **[V3 单卡算子深度融合与 CUDA Graph 优化剖析](docs/v3_kernel_fusion_and_cuda_graph.md)**
- **[V4 单机多卡张量并行 (TP) 与通信消除精要](docs/v4_tensor_parallelism.md)**
- **[V5 MoE 混合专家模型与专家并行 (EP) 深度指南](docs/v5_moe_and_expert_parallelism.md)**
- **[完整 V0~V5 研发规格说明书 (ROADMAP)](ROADMAP.md)**

### 高清架构图解 (SVG 矢量插图全集)
- [V0 Transformer Decoder 算子流水全景图](docs/assets/v0_architecture.svg)
- [V0 极简自回归生成与 O(N²) 冗余分析图](docs/assets/v0_autoregressive_loop.svg)
- [V1 传统连续预分配 vs PagedAttention 显存碎片对比图](docs/assets/v1_paged_attention_concept.svg)
- [V1 Prefill 与 Decode 两阶段执行流水图](docs/assets/v1_prefill_decode_pipeline.svg)
- [V2 传统静态批处理气泡 vs 连续批处理动态流对比图](docs/assets/v2_static_vs_continuous_batching.svg)
- [V2 调度器状态机与连续批处理迭代流图](docs/assets/v2_scheduler_state_machine.svg)
- [V3 算子融合前后 GPU 显存带宽与访存往返对比图](docs/assets/v3_kernel_fusion_memory_bandwidth.svg)
- [V3 CUDA Graph 录制回放与 CPU 发射开销彻底消除机制图](docs/assets/v3_cuda_graph_decode_pipeline.svg)
- [V4 Megatron 张量并行列并行与行并行切分架构图](docs/assets/v4_tensor_parallelism_megatron.svg)
- [V4 单机多卡分布式 Worker 进程与通信架构图](docs/assets/v4_multi_process_worker_architecture.svg)
- [V5 MoE 门控路由网络与“专家找 Token”计算重排机制图](docs/assets/v5_moe_routing_and_grouped_gemm.svg)
- [V5 专家并行 (EP) 与两次 All-to-All 跨卡流水图](docs/assets/v5_expert_parallelism_all_to_all.svg)
