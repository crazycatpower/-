# -*- coding: utf-8 -*-
"""檢查 regulations 資料表是否已匯入。"""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
from db import db_conn

with db_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM regulations")
        print("總筆數：", cur.fetchone()["n"])

        print("\n各來源段數：")
        cur.execute("SELECT source, COUNT(*) AS c FROM regulations GROUP BY source ORDER BY source")
        for r in cur.fetchall():
            print(f"  - {r['source']}：{r['c']} 段")

        print("\n抽一段「職業安全衛生法」的內容看看：")
        cur.execute(
            "SELECT chunk_index, text FROM regulations WHERE source = %s ORDER BY chunk_index LIMIT 1",
            ("職業安全衛生法.md",),
        )
        row = cur.fetchone()
        if row:
            print(f"  [chunk {row['chunk_index']}] {row['text'][:120]}...")
        else:
            print("  （查無資料）")
