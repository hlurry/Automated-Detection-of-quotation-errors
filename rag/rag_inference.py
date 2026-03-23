"""
RAG推理主脚本
支持两种检索方案的对比实验
"""
import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict

from config import DATASETS, MAX_CONTEXT_CHARS, VALID_LABELS
from utils import (
    load_jsonl, save_jsonl, load_progress, save_progress,
    read_txt_file, extract_claim, get_original_label,
    extract_label_from_output, calculate_metrics, print_metrics
)
from retriever import get_retriever
from llm_client import QwenClient


def get_system_prompt() -> str:
    """获取系统提示词"""
    return """You are an expert in the medical field and are familiar with other scientific fields. You will receive a scientific claim and relevant content from a research paper. Based on the paper content, determine whether the paper's findings SUPPORT, CONTRADICT, or have no clear relationship (NULL) with the claim.

Rules:
- If the paper provides evidence that agrees with the claim, return SUPPORT
- If the paper provides evidence that disagrees with the claim, return CONTRADICT  
- If there is no clear relationship between the paper content and the claim, return NULL

Analyze the evidence briefly, then provide your final answer."""


def run_inference(
    dataset_name: str,
    fold: int,
    method: str,
    base_dir: str = ".",
    resume: bool = True
):
    """
    运行RAG推理
    
    Args:
        dataset_name: 数据集名称 (scifact/smith)
        fold: fold编号 (1-5)
        method: 检索方法 (hybrid/hyde)
        base_dir: 基础目录
        resume: 是否断点续传
    """
    # 获取数据集配置
    dataset_config = DATASETS[dataset_name]
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    # 文件路径
    input_file = os.path.join(dataset_dir, dataset_config["input_pattern"].format(fold))
    output_file = os.path.join(dataset_dir, dataset_config["output_pattern"].format(fold, method))
    progress_file = os.path.join(dataset_dir, f"progress_fold{fold}_{method}.json")
    txt_dir = os.path.join(dataset_dir, dataset_config["txt_dir"])
    
    print(f"\n{'='*60}")
    print(f"RAG推理 - {dataset_name} Fold {fold} - 方法: {method}")
    print(f"{'='*60}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"TXT目录: {txt_dir}")
    
    # 加载数据
    data = load_jsonl(input_file)
    print(f"总样本数: {len(data)}")
    
    # 断点续传
    start_idx = 0
    results = []
    if resume and os.path.exists(output_file):
        results = load_jsonl(output_file)
        start_idx = load_progress(progress_file)
        print(f"断点续传: 从第 {start_idx + 1} 条开始")
    
    if start_idx >= len(data):
        print("所有样本已处理完成")
        return results
    
    # 初始化组件
    print("\n初始化组件...")
    llm_client = QwenClient()
    retriever = get_retriever(method, llm_client)
    system_prompt = get_system_prompt()
    
    print(f"开始推理 (从第 {start_idx + 1} 条到第 {len(data)} 条)...")
    
    # 处理每条数据
    for idx in tqdm(range(start_idx, len(data)), desc=f"Processing {method}"):
        record = data[idx]
        
        try:
            # 提取信息
            user_content = record['messages'][1]['content']
            claim = extract_claim(user_content)
            ref_id = record.get('ref_id', '')
            original_label = get_original_label(record)
            
            # 读取论文全文
            txt_path = os.path.join(txt_dir, f"{ref_id}.txt")
            full_text = read_txt_file(txt_path)
            
            if not full_text:
                print(f"\n警告: 无法读取 {txt_path}")
                result = {
                    **record,
                    'original_label': original_label,
                    'model_output': 'ERROR: Cannot read full text',
                    'predicted_label': 'UNKNOWN',
                    'retrieved_context': '',
                    'is_correct': False,
                    'method': method
                }
            else:
                # 检索相关段落
                retrieved_chunks = retriever.retrieve(claim, full_text)
                context = "\n\n".join(retrieved_chunks)
                
                # 截断上下文
                if len(context) > MAX_CONTEXT_CHARS:
                    context = context[:MAX_CONTEXT_CHARS]
                
                # 调用LLM推理
                model_output = llm_client.verify_claim(system_prompt, claim, context)
                predicted_label = extract_label_from_output(model_output)
                
                # 构建结果
                result = {
                    **record,
                    'original_label': original_label,
                    'model_output': model_output,
                    'predicted_label': predicted_label,
                    'retrieved_context': context[:1000] + '...' if len(context) > 1000 else context,
                    'is_correct': predicted_label == original_label,
                    'method': method
                }
            
            results.append(result)
            
            # 定期保存
            if (idx + 1) % 10 == 0:
                save_jsonl(results, output_file)
                save_progress(progress_file, idx + 1)
                
        except Exception as e:
            print(f"\n错误处理第 {idx + 1} 条: {e}")
            result = {
                **record,
                'original_label': get_original_label(record),
                'model_output': f'ERROR: {str(e)}',
                'predicted_label': 'UNKNOWN',
                'retrieved_context': '',
                'is_correct': False,
                'method': method
            }
            results.append(result)
    
    # 最终保存
    save_jsonl(results, output_file)
    save_progress(progress_file, len(data))
    
    # 计算并打印指标
    metrics = calculate_metrics(results)
    print_metrics(metrics, f"{dataset_name} Fold {fold} - {method}")
    
    return results


def run_all_experiments(dataset_name: str, base_dir: str = "."):
    """
    运行所有实验（两种方法 x 5个fold）
    """
    dataset_config = DATASETS[dataset_name]
    all_results = {}
    
    for method in ["hybrid", "hyde"]:
        method_results = []
        for fold in dataset_config["folds"]:
            results = run_inference(dataset_name, fold, method, base_dir)
            method_results.extend(results)
        
        # 汇总该方法的所有结果
        metrics = calculate_metrics(method_results)
        all_results[method] = {
            'results': method_results,
            'metrics': metrics
        }
        print_metrics(metrics, f"{dataset_name} 全部Fold - {method}")
    
    # 对比两种方法
    print("\n" + "="*60)
    print("方法对比")
    print("="*60)
    for method, data in all_results.items():
        print(f"{method}: 准确率 = {data['metrics']['accuracy']:.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='RAG推理脚本')
    parser.add_argument('--dataset', type=str, default='scifact', 
                        choices=['scifact', 'smith'], help='数据集名称')
    parser.add_argument('--fold', type=int, default=None, 
                        help='Fold编号(1-5)，不指定则运行所有fold')
    parser.add_argument('--method', type=str, default='both',
                        choices=['hybrid', 'hyde', 'both'], help='检索方法')
    parser.add_argument('--base_dir', type=str, default='.', help='基础目录')
    parser.add_argument('--no_resume', action='store_true', help='不使用断点续传')
    
    args = parser.parse_args()
    
    if args.fold is not None:
        # 运行单个fold
        methods = ['hybrid', 'hyde'] if args.method == 'both' else [args.method]
        for method in methods:
            run_inference(args.dataset, args.fold, method, args.base_dir, not args.no_resume)
    else:
        # 运行所有fold
        if args.method == 'both':
            run_all_experiments(args.dataset, args.base_dir)
        else:
            dataset_config = DATASETS[args.dataset]
            for fold in dataset_config["folds"]:
                run_inference(args.dataset, fold, args.method, args.base_dir, not args.no_resume)


if __name__ == "__main__":
    main()
