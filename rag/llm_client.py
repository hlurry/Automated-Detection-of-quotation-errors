"""
LLM客户端模块 - 调用Qwen-Plus API
"""
import time
from openai import OpenAI
from config import QWEN_API_KEY, QWEN_MODEL, QWEN_BASE_URL, API_DELAY

class QwenClient:
    """Qwen API客户端"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL
        )
        self.model = QWEN_MODEL
        self.last_call_time = 0
    
    def _wait_for_rate_limit(self):
        """等待以避免API限流"""
        elapsed = time.time() - self.last_call_time
        if elapsed < API_DELAY:
            time.sleep(API_DELAY - elapsed)
    
    def chat(self, user_message: str, system_message: str = None, max_tokens: int = 500) -> str:
        """
        发送聊天请求
        """
        self._wait_for_rate_limit()
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1  # 低温度以获得更确定的输出
            )
            self.last_call_time = time.time()
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用错误: {e}")
            raise
    
    def verify_claim(self, system_prompt: str, claim: str, context: str) -> str:
        """
        验证claim - 使用检索到的上下文
        """
        user_message = f"""Here is the claim: '{claim}'

Here is the relevant content from the full paper that may help verify this claim:

{context}

Based on the above content from the paper, determine whether the paper SUPPORTS, CONTRADICTS, or has no clear relationship (NULL) with the claim.

Please analyze briefly and then provide your final answer as one of: SUPPORT, CONTRADICT, or NULL."""

        return self.chat(user_message, system_message=system_prompt)
