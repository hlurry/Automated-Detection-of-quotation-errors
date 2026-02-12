"""
示例：从本地配置文件中读取 API Key，并调用某个 API。

实际使用时：
1. 复制 `config/api_keys_example.json` 为 `config/api_keys.json`
2. 在 `config/api_keys.json` 中填入你自己的真实 API Key
3. 确保 `config/api_keys.json` 不会被提交到 GitHub（已在 `.gitignore` 中忽略）
"""

import json
from pathlib import Path
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "api_keys.json"


def load_api_keys() -> Dict[str, Any]:
    """
    从本地 JSON 配置文件中加载所有服务的 API Key。
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"未找到 {CONFIG_PATH}，请先复制 config/api_keys_example.json 为 "
            f"config/api_keys.json 并填写你的真实 API Key。"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_api_key(service_name: str) -> str:
    """
    按服务名称获取对应的 API Key，例如 service_name='openai'。
    """
    data = load_api_keys()
    try:
        return data[service_name]["api_key"]
    except KeyError as e:
        raise KeyError(
            f"在 {CONFIG_PATH} 中未找到服务 '{service_name}' 或其 'api_key' 字段，请检查配置文件。"
        ) from e


if __name__ == "__main__":
    # 示例：获取 openai 的 API Key
    try:
        openai_key = get_api_key("openai")
        print("成功读取 openai API Key（此处仅做演示，不建议直接打印真实 Key）。")
    except Exception as exc:
        print(f"读取 API Key 失败：{exc}")

