# 《手撕 vLLM：从零构建高性能大模型推理引擎》完整研发规划

本项目参考工业级推理系统（vLLM / SGLang）及《从零实现 vLLM》演进路径，采取**循序渐进、版本迭代（V0 -> V5）**的工程化方式，从纯 PyTorch 最小自回归原型出发，逐步演进至支持 PagedAttention、Continuous Batching、Triton 算子优化、单机多卡张量并行（TP）以及 MoE 专家并行（EP）的完整大模型推理引擎。

---

## 目录
1. [技术选型与硬件基准](#1-技术选型与硬件基准)
2. [项目整体目录架构设计](#2-项目整体目录架构设计)
3. [版本阶段性里程碑与实施规划](#3-版本阶段性里程碑与实施规划)
   - [V0：最小可用自回归推理原型](#v0-最小可用自回归推理原型)
   - [V1：KV Cache 与分页显存管理 (PagedAttention)](#v1-kv-cache-与分页显存管理-pagedattention)
   - [V2：连续批处理与并发调度器 (Continuous Batching)](#v2-连续批处理与并发调度器-continuous-batching)
   - [V3：单卡算子深度融合与 CUDA Graph 优化](#v3-单卡算子深度融合与-cuda-graph-优化)
   - [V4：单机多卡张量并行 (Tensor Parallelism)](#v4-单机多卡张量并行-tensor-parallelism)
   - [V5：MoE 架构支持与专家并行 (Expert Parallelism)](#v5-moe-架构支持与专家并行-expert-parallelism)
4. [质量保障与对比验证体系](#4-质量保障与对比验证体系)
5. [推荐开发步骤与实施建议](#5-推荐开发步骤与实施建议)

---

## 1. 技术选型与硬件基准

| 维度 | 选型 | 备注 |
| :--- | :--- | :--- |
| **开发语言** | Python 3.10+ | 核心粘合与控制面逻辑 |
| **基础深度学习框架** | PyTorch >= 2.4 | 基础 Tensor 操作、分布式接口与 CUDA 流支持 |
| **算子开发语言** | Triton >= 3.0 | 算子融合与 GPU 性能极致挖掘 |
| **权重格式** | Safetensors | 零拷贝内存映射 (`mmap`) 高效权重加载 |
| **分词与前端** | Hugging Face `tokenizers` / `transformers` | 仅使用分词与模型配置解析，不使用其推理逻辑 |
| **通信库** | PyTorch Distributed (NCCL) | 用于 V4/V5 的跨卡集合通信与点对点通信 |
| **基准测试模型** | **Qwen2.5-0.5B / 1.5B / 7B**<br>**Qwen2-57B-A14B (MoE)** 或 **DeepSeek-MoE** 精简版 | 0.5B/1.5B 适合单卡快速调试，7B 适合压测，MoE 适合 V5 验证 |

---

## 2. 项目整体目录架构设计

随着版本递进，项目逐步实现**控制层（Controller/Engine）**、**调度层（Scheduler）**、**显存管理层（Cache/Memory）**与**执行层（Model/Worker）**的分离：

```text
toy_vllm/
├── csrc/                      # 可选：C++/CUDA 扩展（若需进一步定制）
├── toy_vllm/
│   ├── __init__.py
│   ├── config.py              # 全局配置类（ModelConfig, CacheConfig, SchedulerConfig）
│   ├── engine/                # 引擎接口层
│   │   ├── __init__.py
│   │   ├── llm_engine.py      # LLMEngine 核心驱动
│   │   └── async_engine.py    # 异步请求包装（可选）
│   ├── core/                  # 调度与显存核心机制
│   │   ├── block_manager.py   # 物理块与逻辑块分配管理（V1）
│   │   ├── prefix_tree.py     # 前缀缓存 Radix Tree（V1）
│   │   └── scheduler.py       # Continuous Batching 调度器（V2）
│   ├── model_runner/          # 模型运行封装
│   │   ├── model_runner.py    # 单卡 Runner 绑定上下文与 CUDA Graph（V3）
│   │   └── worker.py          # 多卡分布式 Worker 进程执行体（V4）
│   ├── models/                # 模型结构定义
│   │   ├── qwen2.py           # Qwen2 / Qwen2.5 稠密模型架构
│   │   └── qwen2_moe.py       # Qwen2-MoE 稀疏模型架构（V5）
│   ├── modules/               # 模型算子积木
│   │   ├── attention.py       # Naive Attention -> PagedAttention
│   │   ├── layernorm.py       # RMSNorm 算子实现
│   │   ├── rotary_embedding.py# RoPE 旋转位置编码实现
│   │   ├── activation.py      # Silu / SwiGLU 激活函数
│   │   ├── sampler.py         # Top-K / Top-P / Temp 采样
│   │   └── linear.py          # ColumnParallel / RowParallel 线性层（V4）
│   ├── kernels/               # Triton 高性能自定义算子
│   │   ├── paged_attention.py # Triton PagedAttention Kernel
│   │   ├── fused_moe.py       # Triton Grouped GEMM / Fused MoE Kernel（V5）
│   │   └── fused_norm.py      # Triton Fused RMSNorm
│   └── distributed/           # 分布式并行与通信抽象（V4/V5）
│       ├── parallel_state.py  # 进程组、rank、world_size 状态机
│       └── communication.py   # All-Reduce / All-to-All 算子封装
├── tests/                     # 单元测试与对齐测试
│   ├── test_rmsnorm.py
│   ├── test_rope.py
│   ├── test_attention.py
│   ├── test_block_allocator.py
│   └── test_correctness.py    # 与 HuggingFace / 官方 vLLM 输出对比
├── benchmarks/                # 性能与延迟基准评测
│   ├── bench_throughput.py
│   └── bench_latency.py
├── examples/                  # 端到端运行示例脚本
│   └── run_demo.py
├── pyproject.toml
└── README.md
```

---

## 3. 版本阶段性里程碑与实施规划

```
  ┌────────────────────────────────────────────────────────┐
  │ V0: 跑通最小自回归推理 (Token-by-Token, 纯 PyTorch 实现) │
  └───────────────────────────┬────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ V1: KV 缓存与分页显存 (PagedAttention, Block Allocator)  │
  └───────────────────────────┬────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ V2: 连续批处理 (Continuous Batching, Chunked Prefill)   │
  └───────────────────────────┬────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ V3: 单卡算子深度优化 (Triton FlashAttn, CUDA Graph)     │
  └───────────────────────────┬────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ V4: 单机多卡张量并行 (TP, NCCL 集合通信, 多进程 Worker)   │
  └───────────────────────────┬────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ V5: MoE 稀疏大模型与专家并行 (EP, Grouped GEMM)        │
  └────────────────────────────────────────────────────────┘
```

---

### V0：最小可用自回归推理原型

#### 1. 目标
不使用任何复杂显存池或算子优化，使用纯 PyTorch 手写 Qwen2.5 结构，加载真实 Hugging Face 权重，完整跑通自回归文本生成与流式输出。

#### 2. 关键任务分解
1. **模型配置与权重解析**：
   - 读取 Hugging Face 的 `config.json`，映射为自定义强类型配置对象 `ModelConfig`。
   - 实现权重加载器：基于 `safetensors.torch.load_file` 实现按层参数键名映射与内存映射加载。
2. **手写核心模块**：
   - `RMSNorm`：实现不包含减均值的均方根归一化。
   - `RotaryEmbedding`：预计算复数/Cos-Sin 表，对 Query 和 Key 执行旋转编码。
   - `Attention`：实现支持 GQA（Grouped Query Attention）的标准自注意力（Prefill 与 Decode 暂用完整序列拼接）。
   - `SwiGLU`：实现门控前馈网络（Gate Proj、Up Proj、SiLU、Down Proj）。
   - `DecoderLayer` 与 `TransformerModel`：组装残差连接与多层 Transformer。
3. **采样器与引擎循环**：
   - 实现 `Sampler`：实现 Logits 温度缩放、Top-K 截断、Top-P (Nucleus) 过滤及多项式采样。
   - 构建 `LLMEngine.generate()`：实现最基本的 `while not eos_token` 自回归循环与流式 Yield。

#### 3. 验收标准
- 针对相同 Prompt，`toy_vllm` 与 Hugging Face 原生 `AutoModelForCausalLM` 在 greedy 模式下输出完全一致。

---

### V1：KV Cache 与分页显存管理 (PagedAttention)

#### 1. 目标
消除 V0 中每一步对整个历史序列重复计算的算力浪费，并解决传统线性 KV Cache 导致的严重显存碎片问题，实现分页式显存管理。

#### 2. 关键任务分解
1. **显存池设计（Block Allocator）**：
   - 定义显存块物理单元：固定大小 `block_size`（通常为 16 或 32 tokens）。
   - 物理显存分配：在 GPU 上一次性开辟预分配的大张量 `key_cache` 和 `value_cache`，形状为 `(num_blocks, num_kv_heads, head_dim, block_size)`。
   - 块分配器 `BlockAllocator`：基于空闲链表/栈管理物理块的申请、释放与引用计数。
2. **逻辑块与物理块映射（Block Table）**：
   - 每个推理请求维护一个 `block_table: List[int]`，记录逻辑 block 对应的物理 block id。
   - 随生成逐步向后扩展，当最后一个 block 满时触发新 block 申请。
3. **前向交互契约（ForwardContext）**：
   - 封装 `ForwardContext` 包含：`block_tables`、`slot_mapping`（当前新 token 对应写入的物理槽位索引）。
4. **前缀缓存（Prefix Caching，进阶）**：
   - 针对系统提示词（System Prompt），利用前缀哈希或 Radix Tree 结构，实现多请求之间的物理块共享与写时复制（CoW）。

#### 3. 验收标准
- 单请求推理时，Decode 阶段只计算当前 1 个 Token 的 Q，并能正确从物理 Block 中抓取历史 KV；
- 长文本生成不再发生显存 OOM 或重复计算。

---

### V2：连续批处理与并发调度器 (Continuous Batching)

#### 1. 目标
支持多并发请求动态进出引擎，解决静态批处理（Static Batching）下“长文本拖慢短文本”的问题，实现毫秒级动态调度。

#### 2. 关键任务分解
1. **请求状态机与生命周期**：
   - 定义请求状态：`WAITING`（待分配）、`RUNNING`（正在推理）、`SWAPPED`（显存不足被换出/抢占）、`FINISHED`（遇到 EOS 或超长）。
2. **调度策略（Scheduler）**：
   - 区分 Prefill 阶段与 Decode 阶段。
   - **动态打包（Dynamic Batching）**：在每次 Forward 迭代时，将处于 RUNNING 状态的多个请求组织进同一个批次执行单步 Decode；同时若显存充足，引入新的 WAITING 请求执行 Prefill。
3. **分块前缀（Chunked Prefill）**：
   - 将超长 Prompt 的 Prefill 拆分为多个小 Chunk（例如 512 tokens），与短请求的 Decode 混合批处理，平滑 GPU 利用率与 TTFT（首字延迟）。
4. **架构解耦**：
   - 严格分离为三层：
     - **调度层（Scheduler）**：负责请求决策与显存块分配；
     - **执行层（ModelRunner）**：负责组装 Batch Tensor 并提交 GPU 执行；
     - **引擎层（LLMEngine）**：暴露 `add_request()`、`step()` 与轮询迭代接口。

#### 3. 验收标准
- 模拟 10 个不同长度的并发请求，短请求能提前完成退出释放资源，长请求平稳持续生成；
- 多请求并发吞吐（Tokens/s）相较单请求线性排队提升 3x 以上。

---

### V3：单卡算子深度融合与 CUDA Graph 优化

#### 1. 目标
解决 PyTorch 原生算子发射 Overhead 高、内存带宽瓶颈（Memory-bound）问题，利用 Triton 和 CUDA Graph 榨干单卡硬件潜力。

#### 2. 关键任务分解
1. **Triton PagedAttention 算子手写**：
   - 编写适用于分页显存布局的 FlashAttention/PagedAttention Triton Kernel。
   - 处理变长序列的 Block Table 间接寻址读取与 Softmax 在线重标度。
2. **融合算子（Kernel Fusion）**：
   - **Fused RMSNorm + Residual**：将上层残差累加与本层 RMSNorm 融合成单个 Kernel，减少一次全局显存写与读。
   - **Fused SwiGLU**：将 Gate 与 Up 投影后的逐元素计算 `(x * sigmoid(x)) * y` 进行原位融合。
3. **CUDA Graph 捕获与回放**：
   - 分析 Decode 阶段的特征：固定 batch_size 下，Tensor 形状与执行路径完全确定。
   - 维护一组预热的分档 Batch 尺寸（例如 `[1, 2, 4, 8, 16, 32]`）。
   - 预先对各尺寸录制 CUDA Graph，在推理时直接 `graph.replay()`，将 CPU Launch 开销降至趋近于 0。

#### 3. 验收标准
- Profile 工具（Nsight Systems / PyTorch Profiler）抓取的执行流中，Decode 阶段 CPU 与 GPU 之间无明显气泡；
- 单卡 Decode 延迟（Inter-Token Latency）显著下降。

---

### V4：单机多卡张量并行 (Tensor Parallelism)

#### 1. 目标
突破单卡显存对模型参数量的限制，使用张量并行（Tensor Parallelism, TP）在单机多卡（如 2卡、4卡、8卡）上并行加速。

#### 2. 关键任务分解
1. **多进程控制面设计**：
   - 主进程负责接收调度请求；
   - 采用多 Worker 进程模型，每个 GPU 绑定一个独立的 Worker，通过进程间事件或队列同步执行步调。
2. **通信抽象与分布式状态管理**：
   - 初始化 `torch.distributed` NCCL 进程组；
   - 封装分布式工具函数：`get_tp_rank()`、`get_tp_world_size()`。
3. **模型切分与分布式算子**：
   - **ColumnParallelLinear（列并行）**：用于 `qkv_proj` 和 `gate_up_proj`，输出按通道切分，无需通信。
   - **RowParallelLinear（行并行）**：用于 `o_proj` 和 `down_proj`，输入按通道切分，输出后执行 `all_reduce` 汇总。
   - **VocabParallelEmbedding（词表并行）**：按词表大小将 Embedding 和 LM Head 进行卡间切分。
4. **多卡权重分割载入**：
   - 权重加载器按 Rank 索引读取或切片 Safetensors 权重分片，避免每张卡将完整大模型加载到 CPU 内存导致 OOM。

#### 3. 验收标准
- 2 卡 / 4 卡 TP 模式下模型输出与单卡保持比特级/高精度对齐；
- 能够无阻碍加载单卡无法容纳的大模型（如 7B/14B 跨卡推理）。

---

### V5：MoE 架构支持与专家并行 (Expert Parallelism)

#### 1. 目标
支持当代前沿的混合专家模型（MoE，如 Qwen2-MoE / DeepSeek-V2 系列架构），实现高效门控路由、Grouped GEMM 优化及多卡专家并行（EP）。

#### 2. 关键任务分解
1. **MoE 核心网络结构**：
   - 实现 Top-K 门控路由网络（Router / Gate），包含 Softmax 归一化与权重计算。
   - 支持共享专家（Shared Expert）与细粒度专家的并行前向累加。
2. **计算模式优化（专家找 Token 与 Grouped GEMM）**：
   - 传统实现（每个专家逐个调用 Linear）极慢且碎片化；
   - 重构输入管线：根据 Router 结果重排 Token，将同一个专家的所有 Token 聚集在一起；
   - 编写 Triton 版本的 Grouped GEMM 算子或批处理 GEMM，实现多专家融合并行计算。
3. **专家并行（Expert Parallelism, EP）**：
   - 在多卡环境下，将不同专家分布在不同 GPU 上；
   - 采用 `all_to_all` 通信原语：
     1. Token 发送阶段：每张卡将 Token 按路由目标分发给对应 GPU；
     2. 专家计算阶段：各 GPU 执行本地专家运算；
     3. 结果回收阶段：再次通过 `all_to_all` 将专家输出发回源 GPU。

#### 3. 验收标准
- MoE 模型前向输出与官方 Hugging Face 保持高精度对齐；
- EP 模式下通信吞吐与计算重叠表现稳定。

---

## 4. 质量保障与对比验证体系

为确保每一步不走弯路，必须建立严格的自动化测试与对齐流程：

1. **数值对齐测试（Correctness Tests）**：
   - 每完成一个模块（RMSNorm, RoPE, Attention, MLP），立即编写单测与 `transformers` 对应模块的输出张量计算 `torch.allclose(atol=1e-3, rtol=1e-3)`。
2. **端到端文本生成一致性**：
   - 固化测试 Prompt 集合与随机种子（Greedy Search），对比完整回复内容。
3. **性能基准测试（Benchmark）**：
   - **TTFT (Time To First Token)**：衡量 Prefill 阶段延迟；
   - **ITL (Inter-Token Latency)**：衡量 Decode 阶段每生成一个 token 的耗时；
   - **Throughput (Tokens/s)**：在 16/32/64 并发下的系统总吞吐。

---

## 5. 推荐开发步骤与实施建议

| 阶段 | 预计周期 | 核心产出物 | 核心收获与面试关注点 |
| :--- | :--- | :--- | :--- |
| **Stage 1 (V0)** | 3~5 天 | 跑通单请求自回归，代码行数约 500 行 | 掌握 Safetensors 解析、自回归生成原理、GQA 与 RoPE 细节 |
| **Stage 2 (V1)** | 5~7 天 | 实现 BlockAllocator、BlockTable 与 Paged KV 机制 | 彻底搞懂操作系统分页在显存中的映射，PagedAttention 动机 |
| **Stage 3 (V2)** | 7~10 天 | 实现 Scheduler、Continuous Batching 动态调度 | 深入理解 Prefill/Decode 混合调度、长短文本负载均衡 |
| **Stage 4 (V3)** | 7~10 天 | 手写 Triton PagedAttention、算子融合、CUDA Graph | 掌握 Triton 编程范式、GPU 显存带宽分析、消除 CPU 开销 |
| **Stage 5 (V4)** | 5~7 天 | 实现多进程分布式 Worker 与 Tensor Parallelism | 掌握 NCCL 通信机制、Row/Col 线性层并行切分原理 |
| **Stage 6 (V5)** | 7~10 天 | 实现 MoE 门控、Grouped GEMM 与 EP 专家并行 | 掌握现代 SOTA 模型核心推理机制、All-to-All 分布式通信 |
