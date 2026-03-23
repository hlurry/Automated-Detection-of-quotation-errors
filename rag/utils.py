"""
RAG项目工具函数
"""
import json
import re
import os
from typing import List, Dict, Tuple, Optional

def load_jsonl(filepath: str) -> List[Dict]:
    """加载jsonl文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """保存jsonl文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_progress(progress_file: str) -> int:
    """加载进度文件，返回已处理的行数"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            return progress.get('processed_count', 0)
    return 0

def save_progress(progress_file: str, processed_count: int):
    """保存进度"""
    with open(progress_file, 'w') as f:
        json.dump({'processed_count': processed_count}, f)

def read_txt_file(filepath: str) -> str:
    """读取txt文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def extract_claim(content: str) -> str:
    """从user content中提取claim"""
    match = re.search(r"Here is the claim: '(.+?)' Here is the abstract:", content, re.DOTALL)
    if match:
        return match.group(1)
    # 备用匹配
    match = re.search(r"Here is the claim: '(.+?)'", content, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def extract_abstract(content: str) -> str:
    """从user content中提取abstract"""
    match = re.search(r"Here is the abstract: '(.+)'$", content, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"Here is the abstract: '(.+)$", content, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"Here is the abstract: (.+)$", content, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def extract_label_from_output(output: str) -> str:
    """从模型输出中提取标签"""
    output_upper = output.upper()
    
    # 优先匹配明确的标签
    if "SUPPORT" in output_upper and "CONTRADICT" not in output_upper:
        return "SUPPORT"
    if "CONTRADICT" in output_upper and "SUPPORT" not in output_upper:
        return "CONTRADICT"
    if "NULL" in output_upper:
        return "NULL"
    
    # 如果同时出现多个标签，尝试找最后出现的
    labels = []
    for label in ["SUPPORT", "CONTRADICT", "NULL"]:
        pos = output_upper.rfind(label)
        if pos != -1:
            labels.append((pos, label))
    
    if labels:
        labels.sort(key=lambda x: x[0], reverse=True)
        return labels[0][1]
    
    return "UNKNOWN"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """将文本分块"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks

def get_original_label(record: Dict) -> str:
    """从记录中获取原始标签"""
    messages = record.get('messages', [])
    for msg in messages:
        if msg.get('role') == 'assistant':
            return msg.get('content', '').strip().upper()
    return "UNKNOWN"

def calculate_metrics(results: List[Dict]) -> Dict:
    """计算评估指标"""
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))
    
    # 按标签统计
    label_stats = {}
    for label in ["SUPPORT", "CONTRADICT", "NULL"]:
        label_results = [r for r in results if r.get('original_label') == label]
        label_correct = sum(1 for r in label_results if r.get('is_correct', False))
        label_stats[label] = {
            'total': len(label_results),
            'correct': label_correct,
            'accuracy': label_correct / len(label_results) if label_results else 0
        }
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'label_stats': label_stats
    }

def print_metrics(metrics: Dict, method_name: str):
    """打印评估指标"""
    print(f"\n{'='*60}")
    print(f"评估结果 - {method_name}")
    print(f"{'='*60}")
    print(f"总样本数: {metrics['total']}")
    print(f"正确数: {metrics['correct']}")
    print(f"准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\n按标签统计:")
    for label, stats in metrics['label_stats'].items():
        print(f"  {label}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")
    print(f"{'='*60}\n")
