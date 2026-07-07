# -*- coding: utf-8 -*-
"""一次性匯入腳本：把 regulations/ 下的 .md/.txt 匯入 PostgreSQL。"""
import sys
import io
from pathlib import Path

# 強制 stdout 用 UTF-8，避免 cp950 終端編碼錯誤
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from db import db_conn
from knowledge_base import (
    init_regulations_table,
    import_regulations_to_db,
    list_regulation_sources,
)


def main() -> int:
    # 1) 測試連線
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                ver = cur.fetchone()
        print(f"[OK] 資料庫連線成功：{list(ver.values())[0][:40]}...")
    except Exception as e:
        print(f"[ERROR] 無法連線資料庫：{e}")
        return 1

    # 2) 建表 + 匯入
    init_regulations_table()
    reg_dir = BASE_DIR / "regulations"
    n = import_regulations_to_db(reg_dir)
    print(f"[OK] 匯入完成，共 {n} 段。")

    # 3) 列出結果
    print("目前 regulations 表內容：")
    for r in list_regulation_sources():
        print(f"  - {r['source']}：{r['chunks']} 段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
