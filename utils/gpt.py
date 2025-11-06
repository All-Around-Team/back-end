from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai import OpenAI
import base64
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OCR_SYSTEM_PROMPT = """
You are a world-class OCR engine.
Your only job is to extract text EXACTLY as it appears in the image.

Rules:
- Keep ALL text exactly as-is
- Preserve whitespace, punctuation, capitalization, line breaks
- Do NOT translate or reformat
- Do NOT answer with apologies or comments
- If text is unreadable or empty, return "<EMPTY>"
"""


async def gpt_ocr(image_bytes: bytes) -> str:
  b64 = base64.b64encode(image_bytes).decode("utf-8")

  def _run() -> str:
    system_msg: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": OCR_SYSTEM_PROMPT,
    }

    image_part: ChatCompletionContentPartImageParam = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }

    text_part: ChatCompletionContentPartTextParam = {
        "type": "text",
        "text": "Extract ONLY the text, nothing else.",
    }

    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": [image_part, text_part],
    }

    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[system_msg, user_msg],
        temperature=0,
        max_tokens=2048,
    )

    text = res.choices[0].message.content or ""
    text = text.strip()

    banned = ["sorry", "cannot", "I'm unable", "I can't", "as an AI"]
    if any(w in text.lower() for w in banned):
      return "<EMPTY>"

    return text if text else "<EMPTY>"

  return await asyncio.to_thread(_run)
