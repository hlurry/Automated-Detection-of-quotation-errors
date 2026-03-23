"""
RAG项目配置文件
"""
import os
import json
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[1]

# API配置 - 复用项目现有的密钥管理
def _load_qwen_api_key():
    """从 config/api_keys.json 读取Qwen密钥"""
    config_path = ROOT_DIR / "config" / "api_keys.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("qwen", {}).get("api_key", "")
    return os.getenv("DASHSCOPE_API_KEY", "")

QWEN_API_KEY = _load_qwen_api_key()
QWEN_MODEL = "qwen-plus"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 检索配置
CHUNK_SIZE = 500          # 每个文本块的字符数
CHUNK_OVERLAP = 100       # 文本块之间的重叠字符数
TOP_K = 6                 # 检索返回的top-k个块
BM25_WEIGHT = 0.5         # BM25在混合检索中的权重
SEMANTIC_WEIGHT = 0.5     # 语义检索在混合检索中的权重

# Embedding模型配置
EMBEDDING_MODEL = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
# 备选: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# 推理配置
MAX_CONTEXT_CHARS = 4000  # 最大上下文字符数
API_DELAY = 0.5           # API调用间隔（秒）
BATCH_SIZE = 10           # 批量处理大小

# 数据集配置
DATASETS = {
    "scifact": {
        "input_pattern": "input_scifact_rag_fold{}.jsonl",
        "output_pattern": "output_scifact_rag_fold{}_{}.jsonl",  # 第二个{}是方案名
        "txt_dir": "scifact_txt",
        "folds": [1, 2, 3, 4, 5]
    },
    "smith": {
        "input_pattern": "input_smith_rag_fold{}.jsonl",
        "output_pattern": "output_smith_rag_fold{}_{}.jsonl",
        "txt_dir": "smith_txt",
        "folds": [1, 2, 3, 4, 5]
    }
}

# 标签
VALID_LABELS = ["SUPPORT", "CONTRADICT", "NULL"]
