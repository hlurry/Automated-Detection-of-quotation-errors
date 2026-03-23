import os
import json
import numpy as np
import re

# 标签设置
LABELS = ["SUPPORT", "CONTRADICT", "NULL"]

def extract_last_assistant_label(messages, label_type="standard"):
    """
    从 messages 中提取最后一个 assistant 的标签
    对于3-shot格式，无论标准还是预测，都取最后一个assistant作为标签
    """
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    
    if not assistant_messages:
        return None
    
    # 3-shot格式：始终取最后一个assistant的内容
    content = assistant_messages[-1].get("content", "")
    
    return extract_label_from_content(content)


def extract_label_from_content(content):
    """
    从模型输出内容中提取标签
    支持多种格式：直接标签、解释后带标签、换行分隔等
    """
    if not content:
        return None
    
    content_upper = content.upper()
    
    # 优先匹配明确的标签（支持SUPPORT/CONTRADICT/NULL）
    # 处理可能带有引号的情况
    patterns = [
        r"['\"\[\(]?SUPPORT['\"\]\)]?",  # 带引号或括号
        r"['\"\[\(]?CONTRADICT['\"\]\)]?",
        r"['\"\[\(]?NULL['\"\]\)]?",
    ]
    
    # 首先尝试精确匹配
    for label in LABELS:
        if label in content_upper:
            # 检查是否是独立的词
            import re
            # 匹配标签前后是边界或非字母字符的情况
            pattern = r"(?:^|[\s\[\(\'\"\`\-])" + label + r"(?:$|[\s\]\)\'\"\`\,\.])"
            if re.search(pattern, content_upper):
                return label
    
    # 如果找不到，尝试找包含这些词的行
    lines = content.split('\n')
    for line in reversed(lines):  # 从最后一行开始找
        line_upper = line.upper()
        for label in LABELS:
            if label in line_upper:
                return label
    
    return None

def evaluate_classification(standard_file, predict_file, fold_num=None):
    """评估模块 - 计算PRF1值"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    standard_path = os.path.join(script_dir, standard_file)
    predict_path = os.path.join(script_dir, predict_file)

    true_labels_list = []
    predicted_labels_list = []
    
    # 检查文件是否存在
    if not os.path.exists(standard_path):
        print(f"错误: 标准文件不存在 - {standard_path}")
        return None
    if not os.path.exists(predict_path):
        print(f"错误: 预测文件不存在 - {predict_path}")
        return None
    
    # 读取数据集
    with open(standard_path, 'r', encoding='utf-8') as f_standard, \
            open(predict_path, 'r', encoding='utf-8') as f_predict:
        
        standard_data = [json.loads(line) for line in f_standard]
        predict_data = [json.loads(line) for line in f_predict]
        
        # 检查数据行数是否匹配
        if len(standard_data) != len(predict_data):
            print(f"警告: 标准文件({len(standard_data)}行)与预测文件({len(predict_data)}行)行数不匹配")
        
        # 逐行匹配
        min_lines = min(len(standard_data), len(predict_data))
        for line_num in range(min_lines):
            standard_line = standard_data[line_num]
            predict_line = predict_data[line_num]
            
            # 提取标准标签（标准答案只有一个assistant）
            standard_tag = extract_last_assistant_label(
                standard_line.get("messages", []), 
                label_type="standard"
            )
            
            # 提取预测标签（3-shot，最后一个才是预测）
            predict_tag = extract_last_assistant_label(
                predict_line.get("messages", []),
                label_type="predict"
            )
            
            # 检查标签
            if not standard_tag:
                print(f"数据集第 {line_num + 1} 行标准标签缺失")
                continue
            if not predict_tag:
                # 打印一些信息帮助调试
                last_assistant = None
                for msg in reversed(predict_line.get("messages", [])):
                    if msg.get("role") == "assistant":
                        last_assistant = msg.get("content", "")
                        break
                print(f"数据集第 {line_num + 1} 行预测标签缺失，内容: {last_assistant[:100] if last_assistant else 'None'}...")
                predict_tag = "UNKNOWN"
            
            true_labels_list.append(standard_tag)
            predicted_labels_list.append(predict_tag)
    
    label_metrics = []
    
    # 对每个标签分别计算
    for label in LABELS:
         
        # 标准数据集中, 与当前标签一致的记为 1, 不一致的记为 0
        binary_true_labels = [1 if tag == label else 0 for tag in true_labels_list]
        # 预测数据集中, 与当前标签一致的记为 1, 不一致的记为 0
        binary_predicted_labels = [1 if tag == label else 0 for tag in predicted_labels_list]

        # 标准数据集和预测数据集均与当前标签一致
        TP = np.sum((np.array(binary_true_labels) == 1) & (np.array(binary_predicted_labels) == 1))
        # 标准数据集和预测数据集均与当前标签不一致
        TN = np.sum((np.array(binary_true_labels) == 0) & (np.array(binary_predicted_labels) == 0))
        # 标准数据集与当前标签不一致, 预测数据集与当前标签一致
        FP = np.sum((np.array(binary_true_labels) == 0) & (np.array(binary_predicted_labels) == 1))
        # 标准数据集与当前标签一致, 预测数据集与当前标签不一致
        FN = np.sum((np.array(binary_true_labels) == 1) & (np.array(binary_predicted_labels) == 0))
         
        # 计算公式
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0

        # 输出分标签结果
        print(f"标签: {label}")
        print(f"  TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print("-" * 70)
        
        # 记录分标签参数
        label_metrics.append((precision, recall, f1))
         
    # 计算总体参数平均值
    avg_precision = np.mean([m[0] for m in label_metrics])
    avg_recall = np.mean([m[1] for m in label_metrics])
    avg_f1 = np.mean([m[2] for m in label_metrics])
    
    # 标准数据集与预测数据集标签一致的记为 1, 否则记为 0
    overall_true_labels = [1 if true_labels_list[i] == predicted_labels_list[i] else 0 for i in range(len(true_labels_list))]
    overall_accuracy = np.mean(overall_true_labels)
    
    # 输出总体结果
    print(f"总体:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print("=" * 70)

def find_dataset_pairs(folder_path):
    """
    自动配对数据集
    规则: 根据 fold 数字匹配，一个 input_fold* 对应多个 output_*_fold*
    """
    input_files = []
    output_files = []
    
    # 获取路径下全部文件名
    for file in os.listdir(folder_path):
        if file.endswith('.jsonl'):
            if file.startswith('input'):
                input_files.append(file)
            elif file.startswith('output'):
                output_files.append(file)
    
    pairs = []
    # 按 fold 数字匹配
    for input_file in input_files:
        # 从 input 文件名中提取 fold 数字
        # input_max_A_test_fold1_3shot.jsonl -> fold1
        input_fold_match = re.search(r'fold(\d+)', input_file)
        if not input_fold_match:
            continue
        input_fold = input_fold_match.group(1)
        
        # 查找所有匹配的 output 文件（相同 fold 数字）
        for output_file in output_files:
            output_fold_match = re.search(r'fold(\d+)', output_file)
            if not output_fold_match:
                continue
            output_fold = output_fold_match.group(1)
            
            # fold 数字匹配则配对
            if input_fold == output_fold:
                pairs.append((input_file, output_file, input_fold))
    
    # 按 fold 数字排序
    pairs.sort(key=lambda x: int(x[2]))
    
    return pairs

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 在当前目录查找所有 input/output 文件
    print(f"在 {script_dir} 中查找数据集...")
    dataset_pairs = find_dataset_pairs(script_dir)
    
    if not dataset_pairs:
        print("未找到任一匹配的数据集")
        print("提示: 请确保当前目录下有 input_*.jsonl 和 output_*.jsonl 文件")
    else:
        print(f"找到 {len(dataset_pairs)} 对匹配的数据集\n")
        
        # 按模型分组: model_name -> list of (fold_num, input_file, output_file)
        model_groups = {}
        for input_file, output_file, fold_num in dataset_pairs:
            # 从 output 文件名提取模型名称
            # output_plus_A_test_fold1_3shot.jsonl -> plus
            model_match = re.search(r'output_(\w+)_A_test_fold', output_file)
            model_name = model_match.group(1) if model_match else "unknown"
            
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append((fold_num, input_file, output_file))
        
        # 对每个模型，按 fold 数字排序
        for model_name in model_groups:
            model_groups[model_name].sort(key=lambda x: int(x[0]))
        
        # 按模型展示结果
        for model_name in sorted(model_groups.keys()):
            folds = model_groups[model_name]
            
            print(f"\n{'#'*80}")
            print(f"# 模型: {model_name.upper()}")
            print(f"# 共 {len(folds)} 个 Fold")
            print(f"{'#'*80}")
            
            # 收集该模型所有 fold 的结果用于整体汇总
            all_true_labels = []
            all_pred_labels = []
            
            for fold_num, input_file, output_file in folds:
                print(f"\n{'*'*70}")
                print(f"* Fold {fold_num}: {input_file} <-> {output_file}")
                print(f"{'*'*70}")
                
                # 评估并收集结果
                script_dir = os.path.dirname(os.path.abspath(__file__))
                standard_path = os.path.join(script_dir, input_file)
                predict_path = os.path.join(script_dir, output_file)
                
                evaluate_classification(standard_path, predict_path)
            
            # 显示该模型的整体汇总
            if all_true_labels:
                print(f"\n{'='*80}")
                print(f"【模型 {model_name.upper()} 整体汇总】")
                print(f"{'='*80}")
                # calculate_and_print_metrics(all_true_labels, all_pred_labels, f"{model_name.upper()} 总计")
        
        print(f"\n{'#'*80}")
        print(f"# 评估完成！")
        print(f"{'#'*80}")