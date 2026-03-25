# vLLM Integration for LongBench

## 功能更新

脚本已添加以下功能：

### 1. **vLLM 自动启动和管理**
- 自动启动 vLLM 服务器
- 自动等待服务器就绪
- 优雅的服务器关闭
- 支持多 GPU 并行配置

### 2. **按 Context Length 分组和重启**
- 将 LongBench 问题按输入文本长度分组
- 每跑完一组就自动 kill 和重启 vLLM
- 有助于清理 GPU 内存和避免碎片化

### 3. **灵活的 vLLM 配置**
- 支持张量并行（Tensor Parallel）
- 支持管道并行（Pipeline Parallel）
- 可配置 GPU 内存利用率
- 最大序列长度设置

## 新增参数

### vLLM 启动参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--base-model-path` | - | **必需**。vLLM 要加载的基础模型路径 |
| `--tensor-parallel` | 3 | 张量并行大小（GPU 数量） |
| `--pipeline-parallel` | 1 | 管道并行大小 |
| `--max-model-len` | None | 模型最大序列长度 |
| `--vllm-port` | 8000 | vLLM 服务器端口 |
| `--gpu-memory-utilization` | 0.9 | GPU 内存利用率 |
| `--auto-restart-vllm` | False | 每跑完一组 context 就重启 vLLM |
| `--vllm-timeout` | 300 | 等待 vLLM 就绪的超时时间（秒） |

### 查询参数（保持不变）

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--api-base` | http://127.0.0.1:8000 | vLLM API 服务器地址 |
| `--api-key` | EMPTY | API Key |
| `--model` | qwen3_1.7b_eagle3_1k_128k | 服务的模型名称 |
| `--bench-name` | longbench_ctx_sampled_1k_128k | 基准测试名称 |
| `--answer-file` | - | **必需**。输出结果文件路径 |
| `--max-new-token` | 256 | 最多生成的 token 数 |
| `--temperature` | 0.0 | 采样温度 |
| `--request-timeout` | 600 | 请求超时时间（秒） |

## 使用示例

### 基础用法（带自动重启）
```bash
python scripts/run_longbench_vllm_qwen3.py \
  --base-model-path Qwen/Qwen3-1.7B \
  --tensor-parallel 3 \
  --auto-restart-vllm \
  --answer-file outputs/result.jsonl
```

### 完整示例（3x H100，TP=3）
```bash
python scripts/run_longbench_vllm_qwen3.py \
  --base-model-path Qwen/Qwen3-1.7B \
  --tensor-parallel 3 \
  --pipeline-parallel 1 \
  --max-model-len 131072 \
  --vllm-port 8000 \
  --gpu-memory-utilization 0.9 \
  --auto-restart-vllm \
  --vllm-timeout 300 \
  --answer-file outputs/qwen3_1.7b_vllm_longbench.jsonl \
  --max-new-token 256 \
  --temperature 0.0
```

### 不自动重启（仅启动一次 vLLM）
```bash
python scripts/run_longbench_vllm_qwen3.py \
  --base-model-path Qwen/Qwen3-1.7B \
  --tensor-parallel 3 \
  --answer-file outputs/result.jsonl
```

## 工作流程

1. **启动 vLLM**（如果 `--auto-restart-vllm` 被指定）
   - 使用指定的模型路径和 GPU 配置启动 vLLM
   - 等待服务器就绪（最多等待 `--vllm-timeout` 秒）

2. **按 Context Length 分组**
   - 读取所有问题
   - 按输入文本长度分组（以 1000 token 为单位）
   - 打印分组统计信息

3. **处理每组问题**
   - 对每组中的问题进行推理
   - 记录结果到输出文件

4. **重启 vLLM**（每组结束后）
   - 如果启用了 `--auto-restart-vllm`
   - 优雅停止当前 vLLM 进程
   - 启动新的 vLLM 进程
   - 等待新进程就绪

5. **清理**
   - 最后优雅停止 vLLM 进程

## 输出格式

保存的 JSONL 文件包含：
```json
{
  "question_id": 0,
  "answer_id": "unique_id",
  "model_id": "qwen3_1.7b",
  "choices": [{
    "index": 0,
    "turns": ["answer1", "answer2"],
    "new_tokens": [256, 128],
    "wall_time": [1.5, 0.8]
  }],
  "tstamp": 1234567890.0
}
```

## 调试技巧

### 查看 vLLM 进程日志
```bash
# 如果需要查看 vLLM 输出，可以修改脚本中的 subprocess.PIPE 为 None
# 这样会在控制台显示 vLLM 的日志
```

### 手动启动 vLLM（用于测试）
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --tensor-parallel-size 3 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9
```

### 杀死残留的 vLLM 进程
```bash
# 如果脚本异常退出，可能有残留的 vLLM 进程
pkill -f "vllm.entrypoints"
```

## 注意事项

1. **GPU 要求**：
   - 默认配置需要 3 张 GPU
   - 如需单 GPU，设置 `--tensor-parallel 1`

2. **内存管理**：
   - 每组重启可以清理 GPU 内存
   - 观察内存使用，可能需要调整 `--gpu-memory-utilization`

3. **超时设置**：
   - 对于大规模模型，可能需要增加 `--vllm-timeout`
   - 对于长上下文，增加 `--request-timeout`

4. **依赖包**：
   ```bash
   pip install vllm
   pip install psutil  # 用于进程管理
   ```
