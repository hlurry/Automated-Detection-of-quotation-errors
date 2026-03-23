"""
批量调用 Qwen 对话模型的脚本。

密钥管理方式：
- 请在项目根目录的 config/api_keys.json 中为 "qwen" 配置密钥；
- 本脚本通过公共工具函数读取该密钥，不在代码中硬编码任何密钥。
"""

import os
import json
import glob
import sys
import time
import logging
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def _load_keys():
    """从 config/api_keys.json 读取所有密钥。"""
    config_path = ROOT_DIR / "config" / "api_keys.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"未找到 {config_path}，请参考 config/api_keys_example.json 创建并填写你的密钥。"
        )
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_qwen_secret() -> str:
    """获取 Qwen 对话服务的密钥。"""
    data = _load_keys()
    try:
        return data["qwen"]["api_key"]
    except KeyError as e:
        raise KeyError("在 config/api_keys.json 中未找到 'qwen.api_key' 配置") from e


# 日志文件
LOG_FILE = SCRIPT_DIR / "Error.txt"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _build_client() -> OpenAI:
    """构建 Qwen 兼容模式客户端。"""
    secret = _get_qwen_secret()
    return OpenAI(
        api_key=secret,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _find_user_message(messages):
    """获取 user 内容。"""
    return next((msg for msg in messages if msg["role"] == "user"), None)


def _parse_run_time(time_input: str):
    """解析定时指令, 按时分秒填写, 使用空格分隔。"""
    now = datetime.now()
    parts = time_input.split()
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    second = int(parts[2]) if len(parts) > 2 else 0

    target_time = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if target_time <= now:
        target_time += timedelta(days=1)
        is_tomorrow = True
    else:
        is_tomorrow = False
    return target_time, is_tomorrow


def create_empty_output_file(input_file: str, output_file: str) -> None:
    """根据输入 jsonl 创建对应的空输出文件。

    约定：
    - 输入文件中，一条样本通常形如：
        [system,
         user_example_1, assistant_example_1,
         user_example_2, assistant_example_2,
         user_example_3, assistant_example_3,
         user_target,     assistant_target]
    - 我们希望在调用模型时：
        * 保留 3 个示例的 user / assistant（作为 few-shot 提示）；
        * 仅将最后一个 assistant 视为“待预测标签”，在输入文件中清空其 content。
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        input_data = [json.loads(line) for line in infile]

    output_data = []
    for item in input_data:
        messages = item.get("messages", [])
        # 复制所有消息，然后仅清空“最后一个 assistant” 的 content，
        # 这样可以保留前面 few-shot 示例中的 assistant 标签。
        new_messages = [dict(msg) for msg in messages]

        assistant_indices = [
            idx for idx, msg in enumerate(new_messages) if msg.get("role") == "assistant"
        ]
        if assistant_indices:
            last_idx = assistant_indices[-1]
            new_messages[last_idx]["content"] = ""

        output_data.append({"messages": new_messages})

    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in output_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")


def estimate_file(input_file: str, output_file: str, input_files_dict) -> None:
    """仅估算还有多少条数据待处理（不再计算 token 用量）。"""
    input_filename = os.path.basename(input_file)

    # 根据文件名设定模型：包含 plus 用 Plus，否则用 Max
    model_name = "Qwen-Plus" if "plus" in input_filename else "Qwen-Max"

    with open(input_file, "r", encoding="utf-8") as infile:
        input_data = [json.loads(line) for line in infile]

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            processed_data = [json.loads(line) for line in outfile]
    else:
        processed_data = [{"messages": item["messages"]} for item in input_data]

    remaining_items = 0
    for item in processed_data:
        assistant_list = [msg for msg in item["messages"] if msg["role"] == "assistant"]
        last_assistant = assistant_list[-1] if assistant_list else None
        if not last_assistant or last_assistant["content"] == "":
            remaining_items += 1

    input_files_dict[input_filename] = remaining_items

    if remaining_items == 0:
        print(f"{input_filename} 已处理完成")
        return

    print(f"文件: {input_filename}")
    print(f"模型名称: {model_name}")
    print(f"剩余待处理条目: {remaining_items}")


def process_file(input_file: str, output_file: str, input_files_dict) -> None:
    """顺序处理单个输入文件。"""
    client = _build_client()
    input_filename = os.path.basename(input_file)
    remaining_items = input_files_dict[input_filename]
    error_count = 0

    # 根据文件名设定模型：包含 plus 用 Plus，否则用 Max
    model_id = "qwen-plus-2025-04-28" if "plus" in input_filename else "qwen-max"

    with open(input_file, "r", encoding="utf-8") as infile:
        input_data = [json.loads(line) for line in infile]

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            processed_data = [json.loads(line) for line in outfile]
    else:
        processed_data = [{"messages": item["messages"]} for item in input_data]

    processing_times = deque(maxlen=10)

    for i, item in enumerate(processed_data):
        assistant_list = [msg for msg in item["messages"] if msg["role"] == "assistant"]
        last_assistant = assistant_list[-1] if assistant_list else None
        if last_assistant and last_assistant["content"] != "":
            continue

        try:
            start_time = time.time()
            # 发送给模型的消息：
            # - 包含 system 和所有 few-shot 示例的 user/assistant；
            # - 去掉最后一个 assistant（其 content 在 create_empty_output_file 中已被清空）。
            messages = item["messages"]
            if messages and messages[-1].get("role") == "assistant":
                api_messages = messages[:-1]
            else:
                api_messages = messages

            response = client.chat.completions.create(
                model=model_id,
                messages=api_messages,
            )

            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            assistant_content = response.choices[0].message.content

            if last_assistant:
                last_assistant["content"] = assistant_content
            else:
                item["messages"].append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    }
                )

            with open(output_file, "w", encoding="utf-8") as outfile:
                for row in processed_data:
                    json.dump(row, outfile, ensure_ascii=False)
                    outfile.write("\n")

            remaining_items -= 1
            input_files_dict[input_filename] = remaining_items

            if len(processing_times) < 10:
                estimated_time_display = "--:--"
                avg_processing_time = (
                    sum(processing_times) / len(processing_times)
                    if processing_times
                    else 0
                )
            else:
                avg_processing_time = sum(processing_times) / len(processing_times)
                estimated_time_left = avg_processing_time * remaining_items
                estimated_completion_time = datetime.now() + timedelta(
                    seconds=estimated_time_left
                )
                estimated_time_display = estimated_completion_time.strftime("%H:%M")

            progress = (
                f"正在处理: {input_filename} 的第 {i + 1} / {len(processed_data)} 条数据, "
                f"预估 {estimated_time_display} 完成, "
                f"当前平均处理时间 {avg_processing_time:.2f} 秒"
            )
            error_count = 0
            sys.stdout.write("\r" + progress)
            sys.stdout.flush()

        except Exception as e:  # noqa: BLE001
            error_msg = f"处理 {input_filename} 的第 {i + 1} 条数据时出错: {str(e)}"
            error_count += 1
            logging.error(error_msg)
            if error_count == 1:
                print(f"\n{error_msg}")
            else:
                print(f"{error_msg} (累计错误 {error_count} 次)")
            if error_count > 10:
                time.sleep(60)
            if last_assistant:
                last_assistant["content"] = ""
            else:
                item["messages"].append({"role": "assistant", "content": ""})

    print()


def main():
    input_files = glob.glob(str(SCRIPT_DIR / "input*.jsonl"))
    input_files_dict = {os.path.basename(f): os.path.getsize(f) for f in input_files}

    for input_file in input_files:
        output_file = input_file.replace("input", "output")
        if not os.path.exists(output_file):
            create_empty_output_file(input_file, output_file)
        estimate_file(input_file, output_file, input_files_dict)
        print()

    while any(value != 0 for value in input_files_dict.values()):
        user_input = input("是否继续处理 (y/n/time): ").lower()
        if user_input == "y":
            print("开始运行")
            break
        elif user_input == "n":
            print("取消处理")
            sys.exit()
        elif user_input == "time":
            time_input = input("定时任务启动时间: ")
            target_time, is_tomorrow = _parse_run_time(time_input)
            day_str = "次日" if is_tomorrow else "今日"
            print(f"已设定{day_str} {target_time.strftime('%H:%M:%S')} 开始运行")
            time_to_wait = (target_time - datetime.now()).total_seconds()
            time.sleep(time_to_wait)
            print("定时任务启动")
            break
        else:
            print("无效输入, 请重新输入")

    while any(value != 0 for value in input_files_dict.values()):
        for input_file in input_files:
            file_name = os.path.basename(input_file)
            if input_files_dict[file_name] != 0:
                output_file = input_file.replace("input", "output")
                process_file(input_file, output_file, input_files_dict)


if __name__ == "__main__":
    main()

