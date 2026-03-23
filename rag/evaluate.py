"""
评估脚本 - 对比两种检索方法的效果
包含Precision, Recall, F1和Accuracy指标
支持单个fold和整体评估
"""
import os
import json
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict

from config import DATASETS
from utils import load_jsonl


# 标签列表
LABELS = ["SUPPORT", "CONTRADICT", "NULL"]


def calculate_prf1(results: List[Dict]) -> Dict:
    """
    计算Precision, Recall, F1和Accuracy
    
    返回:
    {
        'accuracy': float,
        'macro_precision': float,
        'macro_recall': float,
        'macro_f1': float,
        'weighted_f1': float,
        'per_label': {
            'SUPPORT': {'precision': float, 'recall': float, 'f1': float, 'support': int},
            'CONTRADICT': {...},
            'NULL': {...}
        }
    }
    """
    # 统计每个标签的TP, FP, FN
    tp = defaultdict(int)  # True Positive
    fp = defaultdict(int)  # False Positive
    fn = defaultdict(int)  # False Negative
    support = defaultdict(int)  # 每个标签的真实样本数
    
    total = len(results)
    correct = 0
    
    for r in results:
        true_label = r.get('original_label', 'UNKNOWN').upper()
        pred_label = r.get('predicted_label', 'UNKNOWN').upper()
        
        # 统计support
        if true_label in LABELS:
            support[true_label] += 1
        
        if true_label == pred_label:
            correct += 1
            if true_label in LABELS:
                tp[true_label] += 1
        else:
            if true_label in LABELS:
                fn[true_label] += 1
            if pred_label in LABELS:
                fp[pred_label] += 1
    
    # 计算每个标签的P, R, F1
    per_label = {}
    for label in LABELS:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_label[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support[label]
        }
    
    # 计算Macro平均
    macro_precision = sum(per_label[l]['precision'] for l in LABELS) / len(LABELS)
    macro_recall = sum(per_label[l]['recall'] for l in LABELS) / len(LABELS)
    macro_f1 = sum(per_label[l]['f1'] for l in LABELS) / len(LABELS)
    
    # 计算Weighted F1
    total_support = sum(support[l] for l in LABELS)
    if total_support > 0:
        weighted_f1 = sum(per_label[l]['f1'] * support[l] for l in LABELS) / total_support
    else:
        weighted_f1 = 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_label': per_label
    }


def print_metrics_table(metrics: Dict, title: str):
    """打印格式化的指标表格"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # 总体指标
    print(f"\n【总体指标】")
    print(f"  样本数: {metrics['total']}")
    print(f"  正确数: {metrics['correct']}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    
    # 每个标签的指标
    print(f"\n【按标签统计】")
    print(f"  {'标签':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    for label in LABELS:
        stats = metrics['per_label'][label]
        print(f"  {label:<12} {stats['precision']:>10.4f} {stats['recall']:>10.4f} {stats['f1']:>10.4f} {stats['support']:>10}")
    print(f"{'='*80}")


def evaluate_dataset(dataset_name: str, base_dir: str = "."):
    """评估整个数据集的两种方法，包含PRF1和ACC"""
    dataset_config = DATASETS[dataset_name]
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    print(f"\n{'#'*80}")
    print(f"# 评估数据集: {dataset_name.upper()}")
    print(f"{'#'*80}")
    
    all_metrics = {}
    
    for method in ["hybrid", "hyde"]:
        method_name = "混合检索 (Hybrid)" if method == "hybrid" else "假设生成检索 (HyDE)"
        print(f"\n\n{'*'*80}")
        print(f"* 方法: {method_name}")
        print(f"{'*'*80}")
        
        method_results = []
        fold_metrics_list = []
        
        # 评估每个fold
        for fold in dataset_config["folds"]:
            output_file = os.path.join(
                dataset_dir, 
                dataset_config["output_pattern"].format(fold, method)
            )
            
            if os.path.exists(output_file):
                results = load_jsonl(output_file)
                fold_metrics = calculate_prf1(results)
                fold_metrics_list.append(fold_metrics)
                method_results.extend(results)
                
                # 打印单个fold的指标
                print_metrics_table(fold_metrics, f"Fold {fold} - {method_name}")
            else:
                print(f"\n⚠️  Fold {fold}: 文件不存在 - {output_file}")
        
        # 计算并打印整体指标
        if method_results:
            total_metrics = calculate_prf1(method_results)
            all_metrics[method] = total_metrics
            print_metrics_table(total_metrics, f"【整体汇总】{method_name} - 全部 {len(dataset_config['folds'])} 个Fold")
    
    # 方法对比
    if len(all_metrics) == 2:
        print(f"\n\n{'#'*80}")
        print(f"# 方法对比总结")
        print(f"{'#'*80}")
        
        print(f"\n{'指标':<20} {'Hybrid':>15} {'HyDE':>15} {'差异':>15}")
        print(f"{'-'*65}")
        
        hybrid = all_metrics['hybrid']
        hyde = all_metrics['hyde']
        
        metrics_to_compare = [
            ('Accuracy', 'accuracy'),
            ('Macro Precision', 'macro_precision'),
            ('Macro Recall', 'macro_recall'),
            ('Macro F1', 'macro_f1'),
            ('Weighted F1', 'weighted_f1')
        ]
        
        for name, key in metrics_to_compare:
            h_val = hybrid[key]
            y_val = hyde[key]
            diff = h_val - y_val
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            print(f"{name:<20} {h_val:>15.4f} {y_val:>15.4f} {diff_str:>15}")
        
        # 结论
        print(f"\n【结论】")
        if hybrid['macro_f1'] > hyde['macro_f1'] + 0.01:
            print(f"  ✅ 混合检索(Hybrid)在Macro F1上优于假设生成检索(HyDE)")
        elif hyde['macro_f1'] > hybrid['macro_f1'] + 0.01:
            print(f"  ✅ 假设生成检索(HyDE)在Macro F1上优于混合检索(Hybrid)")
        else:
            print(f"  ⚖️  两种方法效果相近")
    
    return all_metrics


def generate_report_file(dataset_name: str, base_dir: str = "."):
    """生成评估报告文件"""
    dataset_config = DATASETS[dataset_name]
    dataset_dir = os.path.join(base_dir, dataset_name)
    report_file = os.path.join(dataset_dir, f"evaluation_report_{dataset_name}.txt")
    
    # 收集所有结果
    all_results = {'hybrid': [], 'hyde': []}
    fold_results = {'hybrid': {}, 'hyde': {}}
    
    for method in ["hybrid", "hyde"]:
        for fold in dataset_config["folds"]:
            output_file = os.path.join(
                dataset_dir,
                dataset_config["output_pattern"].format(fold, method)
            )
            if os.path.exists(output_file):
                results = load_jsonl(output_file)
                fold_results[method][fold] = results
                all_results[method].extend(results)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"RAG检索方法评估报告 - {dataset_name.upper()}\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for method in ["hybrid", "hyde"]:
            method_name = "混合检索 (Hybrid)" if method == "hybrid" else "假设生成检索 (HyDE)"
            f.write(f"\n{'='*80}\n")
            f.write(f"方法: {method_name}\n")
            f.write(f"{'='*80}\n\n")
            
            # 每个fold的指标
            f.write("【各Fold指标】\n")
            f.write(f"{'Fold':<8} {'Acc':>8} {'M-P':>8} {'M-R':>8} {'M-F1':>8} {'W-F1':>8} {'样本数':>8}\n")
            f.write("-" * 60 + "\n")
            
            for fold in dataset_config["folds"]:
                if fold in fold_results[method]:
                    metrics = calculate_prf1(fold_results[method][fold])
                    f.write(f"Fold {fold:<3} {metrics['accuracy']:>8.4f} {metrics['macro_precision']:>8.4f} "
                           f"{metrics['macro_recall']:>8.4f} {metrics['macro_f1']:>8.4f} "
                           f"{metrics['weighted_f1']:>8.4f} {metrics['total']:>8}\n")
            
            # 整体指标
            if all_results[method]:
                total_metrics = calculate_prf1(all_results[method])
                f.write("-" * 60 + "\n")
                f.write(f"{'总计':<8} {total_metrics['accuracy']:>8.4f} {total_metrics['macro_precision']:>8.4f} "
                       f"{total_metrics['macro_recall']:>8.4f} {total_metrics['macro_f1']:>8.4f} "
                       f"{total_metrics['weighted_f1']:>8.4f} {total_metrics['total']:>8}\n")
                
                # 每个标签的详细指标
                f.write(f"\n【按标签统计】\n")
                f.write(f"{'标签':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
                f.write("-" * 52 + "\n")
                for label in LABELS:
                    stats = total_metrics['per_label'][label]
                    f.write(f"{label:<12} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
                           f"{stats['f1']:>10.4f} {stats['support']:>10}\n")
        
        # 方法对比
        if all_results['hybrid'] and all_results['hyde']:
            f.write(f"\n\n{'='*80}\n")
            f.write("方法对比\n")
            f.write(f"{'='*80}\n\n")
            
            hybrid_metrics = calculate_prf1(all_results['hybrid'])
            hyde_metrics = calculate_prf1(all_results['hyde'])
            
            f.write(f"{'指标':<20} {'Hybrid':>12} {'HyDE':>12} {'差异':>12}\n")
            f.write("-" * 60 + "\n")
            
            for name, key in [('Accuracy', 'accuracy'), ('Macro Precision', 'macro_precision'),
                             ('Macro Recall', 'macro_recall'), ('Macro F1', 'macro_f1'),
                             ('Weighted F1', 'weighted_f1')]:
                h_val = hybrid_metrics[key]
                y_val = hyde_metrics[key]
                diff = h_val - y_val
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                f.write(f"{name:<20} {h_val:>12.4f} {y_val:>12.4f} {diff_str:>12}\n")
    
    print(f"\n📄 评估报告已保存到: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description='评估RAG推理结果 - 包含PRF1和ACC指标')
    parser.add_argument('--dataset', type=str, default='scifact',
                        choices=['scifact', 'smith'], help='数据集名称')
    parser.add_argument('--base_dir', type=str, default='.', help='基础目录')
    parser.add_argument('--report', action='store_true', help='生成评估报告文件')
    
    args = parser.parse_args()
    
    evaluate_dataset(args.dataset, args.base_dir)
    
    if args.report:
        generate_report_file(args.dataset, args.base_dir)


if __name__ == "__main__":
    main()
