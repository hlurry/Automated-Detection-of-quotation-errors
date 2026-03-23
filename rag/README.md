# RAG项目 - 科学声明验证

本项目使用RAG（检索增强生成）技术，根据论文全文验证科学声明（claim）的正确性。

## 项目结构

```
rag/
├── config.py              # 配置文件
├── utils.py               # 工具函数
├── retriever.py           # 检索模块（两种方案）
├── llm_client.py          # LLM客户端
├── rag_inference.py       # 主推理脚本
├── evaluate.py            # 评估脚本
├── requirements.txt       # 依赖
├── scifact/               # SciFact数据集
│   ├── input_scifact_rag_fold*.jsonl
│   ├── output_scifact_rag_fold*_*.jsonl
│   └── scifact_txt/
└── smith/                 # Smith数据集
    ├── input_smith_rag_fold*.jsonl
    ├── output_smith_rag_fold*_*.jsonl
    └── smith_txt/
```

## 两种检索方案

### 方案A: 混合检索 (hybrid)
- **关键词提取**: 从claim中提取关键实体/概念
- **BM25检索**: 基于关键词的传统检索
- **语义检索**: 基于PubMedBERT的向量检索
- **RRF融合**: 使用Reciprocal Rank Fusion合并结果

**优点**: 能同时捕获关键词匹配和语义相似的内容

### 方案B: 假设生成检索 (hyde)
- **假设生成**: 用LLM生成支持假设和反驳假设
- **多查询检索**: 分别用原始claim、支持假设、反驳假设进行检索
- **结果合并**: 交替合并三种检索结果

**优点**: 能更好地检索到反驳证据

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置API

在 `config.py` 中设置Qwen API密钥：

```python
QWEN_API_KEY = "your-api-key-here"
```

或设置环境变量：

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

## 使用方法

### 1. 运行单个实验

```bash
# 运行scifact数据集的fold1，使用混合检索
python rag_inference.py --dataset scifact --fold 1 --method hybrid

# 运行scifact数据集的fold1，使用假设生成检索
python rag_inference.py --dataset scifact --fold 1 --method hyde

# 同时运行两种方法
python rag_inference.py --dataset scifact --fold 1 --method both
```

### 2. 运行所有fold

```bash
# 运行scifact数据集的所有fold，两种方法
python rag_inference.py --dataset scifact --method both

# 运行smith数据集
python rag_inference.py --dataset smith --method both
```

### 3. 评估结果

```bash
# 评估scifact数据集
python evaluate.py --dataset scifact

# 生成详细对比报告
python evaluate.py --dataset scifact --report
```

## 输出文件格式

输出的jsonl文件每行包含：

```json
{
  "messages": [...],              // 原始messages
  "ref_id": "7098463",            // 论文ID
  "original_label": "SUPPORT",    // 原始标签
  "model_output": "...",          // 模型完整输出
  "predicted_label": "SUPPORT",   // 提取的预测标签
  "retrieved_context": "...",     // 检索到的上下文
  "is_correct": true,             // 是否正确
  "method": "hybrid"              // 使用的方法
}
```

## 断点续传

程序支持断点续传，如果中断可以直接重新运行，会从上次中断的位置继续。

如需重新开始，使用 `--no_resume` 参数：

```bash
python rag_inference.py --dataset scifact --fold 1 --method hybrid --no_resume
```

## 配置参数

在 `config.py` 中可以调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 500 | 文本块大小 |
| CHUNK_OVERLAP | 100 | 文本块重叠 |
| TOP_K | 6 | 检索返回的块数 |
| MAX_CONTEXT_CHARS | 4000 | 最大上下文字符数 |
| API_DELAY | 0.5 | API调用间隔(秒) |
