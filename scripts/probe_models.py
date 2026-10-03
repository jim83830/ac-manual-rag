"""檢查 .env 設定的模型是否可用。執行：docker compose run --rm app python scripts/probe_models.py"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rag.config import ConfigError, load_settings  # noqa: E402
from rag.llm import LLMError, NvidiaClient  # noqa: E402


def sample_image() -> bytes:
    image = Image.new("RGB", (640, 200), "white")
    ImageDraw.Draw(image).text((30, 60), "PROBE 4729", fill="black", font=ImageFont.load_default(size=64))
    buffer = BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def check(name, fn) -> bool:
    try:
        result = fn()
        if not result:
            print(f"❌ {name}：回傳空白（推理型模型可能把 token 都用在思考上）")
            return False
        print(f"✅ {name}：{result}")
        return True
    except LLMError as exc:
        print(f"❌ {name}：{exc}")
        return False


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"設定錯誤：{exc}")
        return 1
    llm = NvidiaClient(settings)
    try:
        available = set(llm.list_models())
    except LLMError as exc:
        print(f"❌ 無法列出模型（API key 是否正確？）：{exc}")
        return 1

    for label, model in [("視覺", settings.vision_model), ("聊天", settings.chat_model), ("Embedding", settings.embed_model)]:
        print(f"{'✅' if model in available else '⚠️ 不在模型清單'} {label}模型 {model}")
    vision_candidates = sorted(m for m in available if any(k in m.lower() for k in ("vision", "vl", "vlm")))
    print("清單中的視覺模型候選：", ", ".join(vision_candidates) or "（找不到）")

    results = [
        check("聊天", lambda: llm.chat([{"role": "user", "content": "用一句繁體中文自我介紹"}], max_tokens=80)),
        check("Embedding", lambda: f"維度 {len(llm.embed(['強力運轉怎麼開'], 'query')[0])}"),
        check("Rerank", lambda: llm.rerank("強力運轉怎麼開", ["定時關機的設定方法", "運轉中按快速鍵可進行強力運轉"])),
        check("視覺", lambda: llm.ocr_page(sample_image(), "圖片裡寫了什麼？只輸出文字。")),
    ]
    print("全部通過 🎉" if all(results) else "有項目失敗：依上面訊息在 .env 換模型後再跑一次")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
