"""用 OpenAI SDK 测试 DeepSeek 连接"""
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or api_key == "sk-your-api-key-here":
    print("❌ 请先在 .env 文件中填入真实的 DeepSeek API Key！")
    exit(1)

from openai import OpenAI
import httpx

urls = [
    "https://api.deepseek.com/v1",
    "https://api.deepseek.com",
]

for url in urls:
    print(f"\n--- 测试: {url} ---")
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=url,
            http_client=httpx.Client(timeout=30.0),
        )
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "说两个字：成功"}],
            max_tokens=10,
        )
        print(f"✅ 成功! 回复: {r.choices[0].message.content}")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
