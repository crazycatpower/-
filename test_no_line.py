# -*- coding: utf-8 -*-
"""
test_no_line.py — 直接呼叫 app.py 正式的事件處理邏輯，但跳過真正的 LINE 平台互動。

跟 simulate_test.py 不同之處：simulate_test.py 只呼叫 analyze_image_bytes 等底層函式，
本檔案改成直接呼叫 app.handle_text() / app._process_image_async()（LINE Webhook 事件
真正會執行的那兩個函式），並用假的 event 物件與假圖片位元組餵給它們。

被攔截、不會真正發生的部分（= 跳過的「LINE 測試」）：
  - app._line_reply / app._line_push   → 改為印出訊息內容，不呼叫 LINE Messaging API
  - app._line_get_content              → 改為讀本地圖片檔，不下載 LINE 使用者上傳的內容
  - Webhook 簽章驗證 / Flask request context → 不經過，因為直接呼叫處理函式本身

用法:
  python test_no_line.py                    # 文字 + 圖片全部案例
  python test_no_line.py --text-only
  python test_no_line.py --image-only --image TEST.jpg
"""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

import app  # noqa: E402  匯入主程式模組（沿用其全部業務邏輯）

SENT: list[tuple[str, str, object]] = []


def _describe(msg) -> str:
    text = getattr(msg, "text", None)
    if text:
        return text[:200].replace("\n", " ")
    return type(msg).__name__


def _fake_reply(reply_token, *messages) -> None:
    for m in messages:
        SENT.append(("reply", reply_token, m))
        print(f"  [reply→{reply_token}] {_describe(m)}")


def _fake_push(to, *messages) -> None:
    for m in messages:
        SENT.append(("push", to, m))
        print(f"  [push→{to}] {_describe(m)}")


def make_event(*, text: str, user_id: str = "test_user", reply_token: str = "fake-reply-token"):
    """建立與 linebot.v3 MessageEvent 屬性相容的假事件（duck typing，不需真實 SDK 物件）。"""
    message = SimpleNamespace(id="fake-msg-id", text=text)
    source = SimpleNamespace(type="user", user_id=user_id, group_id=None, room_id=None)
    return SimpleNamespace(reply_token=reply_token, source=source, message=message)


def run_text_case(label: str, text: str) -> None:
    print(f"\n>>> 文字測試：{label}  輸入=「{text}」")
    app.handle_text(make_event(text=text))


def run_image_case(image_path: Path, prompt: str | None = None) -> None:
    print(f"\n>>> 圖片測試：{image_path.name}")
    source_key = "user:test_user"
    state = app.get_user_state(source_key)
    if prompt:
        state["prompt"] = prompt
    image_bytes = image_path.read_bytes()

    with mock.patch.object(app, "_line_get_content", lambda message_id: image_bytes):
        # 正式流程原本用 threading.Thread 背景執行，這裡同步呼叫以利測試等待結果。
        app._process_image_async(
            user_id="test_user",
            source_key=source_key,
            message_id="fake-msg-id",
            prompt=state.get("prompt") or app.DEFAULT_ANALYZE_PROMPT,
            host_url="http://localhost:5000",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="呼叫 app.py 正式邏輯，跳過真實 LINE 連線")
    parser.add_argument("--image", default="TEST.jpg")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--image-only", action="store_true")
    args = parser.parse_args()

    app.init_audit_db()

    with mock.patch.object(app, "_line_reply", _fake_reply), \
         mock.patch.object(app, "_line_push", _fake_push):

        if not args.image_only:
            run_text_case("說明選單", "說明")
            run_text_case("切換到照片稽核模式", "照片稽核")
            run_text_case("稽核紀錄查詢", "統計")

        if not args.text_only:
            image_path = Path(args.image)
            if not image_path.is_absolute():
                image_path = Path(__file__).parent / image_path
            if image_path.exists():
                run_image_case(image_path)
            else:
                print(f"[SKIP] 找不到圖片：{image_path}")

    print(f"\n共攔截 {len(SENT)} 則訊息（皆未真正送往 LINE，也未連線 LINE Content API）。")


if __name__ == "__main__":
    main()
