"""家人用的問答網頁（Gradio）。啟動：docker compose up app，手機連 http://<電腦IP>:7860"""
from __future__ import annotations

import sys

import gradio as gr

from rag.config import EXAMPLE_QUESTIONS, ConfigError, Settings, load_settings
from rag.llm import NvidiaClient
from rag.service import ask, history_pairs
from rag.store import VectorStore


def build_app(settings: Settings, llm, store: VectorStore) -> gr.Blocks:
    with gr.Blocks(title="家電說明書小幫手") as demo:
        gr.Markdown("# 🏠 家電說明書小幫手")
        if store.count() == 0:
            gr.Markdown("⚠️ 資料庫是空的，請先執行 `docker compose run --rm ingest`")
        gr.Markdown("常見問題（點一下就問）：")
        with gr.Row():
            example_buttons = [gr.Button(q, size="sm") for q in EXAMPLE_QUESTIONS]
        chatbot = gr.Chatbot(type="messages", label="對話", height=420, allow_tags=False)
        gallery = gr.Gallery(label="說明書原頁（點圖可放大）", columns=2, height="auto", object_fit="contain")
        with gr.Accordion("🔍 檢索細節", open=False):
            debug = gr.Markdown()
        with gr.Row():
            box = gr.Textbox(placeholder="輸入問題…", show_label=False, scale=4)
            send = gr.Button("送出", variant="primary", scale=1)
        clear = gr.Button("清除對話")

        def respond(question, messages):
            question = (question or "").strip()
            messages = list(messages or [])
            if not question:
                yield messages, gr.update(), gr.update(), ""
                return
            history = history_pairs(messages)
            base = messages + [{"role": "user", "content": question}]
            for update in ask(llm, store, settings, question, history):
                yield (
                    base + [{"role": "assistant", "content": update.answer}],
                    [str(p) for p in update.pages],
                    update.debug,
                    "",
                )

        outputs = [chatbot, gallery, debug, box]
        send.click(respond, [box, chatbot], outputs)
        box.submit(respond, [box, chatbot], outputs)
        for button in example_buttons:
            button.click(respond, [button, chatbot], outputs)  # 按鈕的值就是它的文字
        clear.click(lambda: ([], [], "", ""), None, outputs)
    return demo


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        sys.exit(f"設定錯誤：{exc}")
    store = VectorStore(settings.chroma_dir)
    demo = build_app(settings, NvidiaClient(settings), store)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(settings.data_dir.resolve())],
    )


if __name__ == "__main__":
    main()
