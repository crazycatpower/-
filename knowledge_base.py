import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class KBSnippet:
    source: str
    text: str
    score: float
    article: str = ""


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= max_chars:
            buf = f"{buf}\n\n{p}"
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    final: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i : i + max_chars].strip())
    return [c for c in final if c]


def _score(query: str, text: str) -> float:
    q = _normalize(query)
    t = _normalize(text)
    if not q or not t:
        return 0.0

    def grams(s: str) -> set[str]:
        s = re.sub(r"\s+", "", s)
        if len(s) <= 1:
            return {s} if s else set()
        return {s[i : i + 2] for i in range(len(s) - 1)}

    qg = grams(q)
    tg = grams(t)
    if not qg or not tg:
        return 0.0
    overlap = len(qg & tg) / max(1, len(qg))
    keywords = list(dict.fromkeys(k for k in re.split(r"[^0-9a-z一-鿿]+", q) if len(k) >= 2))
    hits = sum(1 for k in keywords if k in t)
    bonus = min(0.6, hits * 0.08)
    return float(overlap + bonus)


# =============================================================================
# DB 操作
# =============================================================================

def init_regulations_table() -> None:
    """建立 regulations 資料表（若不存在）。"""
    try:
        from db import db_conn
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS regulations (
                        id BIGSERIAL PRIMARY KEY,
                        source TEXT NOT NULL,
                        article TEXT NOT NULL DEFAULT '',
                        chunk_index INTEGER NOT NULL DEFAULT 0,
                        text TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT ''
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_regulations_source ON regulations(source)")
            conn.commit()
    except Exception as e:
        print(f"[KB] 建立 regulations 表失敗: {e}")


def retrieve_regulations_from_db(query: str, top_k: int = 4) -> list[KBSnippet]:
    if not (query or "").strip():
        return []
    try:
        from db import db_conn

        # 查兩張表：regulations 是 LINE 知識庫上傳功能實際在寫入的表（新資料的來源）；
        # regulations_kb 雖然沒有新資料寫入，但已有 625 筆真實法規條文（含條號，如
        # 職業安全衛生法、營造安全衛生設施標準），內容遠比 regulations 目前的 8 筆
        # 豐富且有條號可引用，不查它等於讓模型每次都只能憑印象猜法規。
        #
        # 不用 ILIKE 關鍵字前篩：_keywords() 沒有中文斷詞能力，一整句「請分析這張施工
        # 現場照片的職安合規性」會變成一個17字的完整字串，幾乎不可能在法規條文裡逐字
        # 出現，導致原本的 ILIKE 前篩實測幾乎永遠篩掉所有真正相關的條文。改成把兩張表
        # （合計約600多筆，量不大）整批撈出來，交給下面已有的 _score()（bigram 重疊比對，
        # 不需要斷詞）在 Python 端排序，跟檔案版 retrieve_regulations_from_files 做法一致。
        sql = """
            SELECT source, article, text FROM regulations
            UNION ALL
            SELECT source_file AS source, article_title AS article, content AS text
            FROM regulations_kb
            LIMIT 2000
        """
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        if not rows:
            return []
        scored = sorted(
            [(r, _score(query, r["text"])) for r in rows],
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            KBSnippet(source=r["source"], article=r.get("article", ""), text=r["text"], score=s)
            for r, s in scored[:top_k]
            if s > 0
        ]
    except Exception as e:
        print(f"[KB] DB 查詢失敗: {e}")
        return []


def retrieve_regulations_from_files(
    query: str, regulations_dir: Path, top_k: int = 4
) -> list[KBSnippet]:
    regulations_dir = Path(regulations_dir)
    if not regulations_dir.exists():
        return []
    candidates: list[KBSnippet] = []
    for p in sorted(regulations_dir.glob("*.md")) + sorted(regulations_dir.glob("*.txt")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        for chunk in _chunk_text(raw):
            s = _score(query, chunk)
            if s <= 0:
                continue
            candidates.append(KBSnippet(source=p.name, text=chunk, score=s))
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:max(0, top_k)]


def retrieve_regulations(
    query: str,
    regulations_dir: Path | None = None,
    top_k: int = 4,
) -> list[KBSnippet]:
    """DB 優先，DB 空時退回檔案。"""
    results = retrieve_regulations_from_db(query, top_k)
    if results:
        return results
    if regulations_dir:
        return retrieve_regulations_from_files(query, regulations_dir, top_k)
    return []


def import_regulations_to_db(regulations_dir: Path) -> int:
    """將 regulations/ 目錄下的 .md/.txt 匯入資料庫，同名 source 先刪後增。"""
    regulations_dir = Path(regulations_dir)
    if not regulations_dir.exists():
        return 0
    try:
        from db import db_conn
        total = 0
        files = sorted(regulations_dir.glob("*.md")) + sorted(regulations_dir.glob("*.txt"))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db_conn() as conn:
            with conn.cursor() as cur:
                for p in files:
                    try:
                        raw = p.read_text(encoding="utf-8", errors="ignore")
                    except OSError as e:
                        print(f"[KB] 無法讀取 {p.name}：{e}")
                        continue
                    chunks = _chunk_text(raw)
                    if not chunks:
                        continue
                    cur.execute("DELETE FROM regulations WHERE source = %s", (p.name,))
                    for i, chunk in enumerate(chunks):
                        cur.execute(
                            "INSERT INTO regulations (source, chunk_index, text, created_at) VALUES (%s,%s,%s,%s)",
                            (p.name, i, chunk, now),
                        )
                        total += 1
            conn.commit()
        return total
    except Exception as e:
        print(f"[KB] 檔案匯入失敗: {e}")
        return 0


def add_regulation_text(source: str, text: str, article: str = "") -> int:
    """直接將一段法規文字切段後存入 DB，回傳插入筆數。"""
    chunks = _chunk_text(text)
    if not chunks:
        return 0
    try:
        from db import db_conn
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM regulations WHERE source = %s", (source,))
                for i, chunk in enumerate(chunks):
                    cur.execute(
                        "INSERT INTO regulations (source, article, chunk_index, text, created_at) VALUES (%s,%s,%s,%s,%s)",
                        (source, article, i, chunk, now),
                    )
            conn.commit()
        return len(chunks)
    except Exception as e:
        print(f"[KB] 文字存入失敗: {e}")
        return 0


def list_regulation_sources() -> list[dict]:
    """列出所有法規來源及筆數。"""
    try:
        from db import db_conn
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source, COUNT(*) AS chunks FROM regulations GROUP BY source ORDER BY source"
                )
                return cur.fetchall()
    except Exception as e:
        print(f"[KB] 列出來源失敗: {e}")
        return []


def format_snippets(snips: list[KBSnippet]) -> str:
    if not snips:
        return ""
    blocks: list[str] = []
    for i, s in enumerate(snips, 1):
        header = f"[{i}] 來源：{s.source}"
        if s.article:
            header += f"｜{s.article}"
        blocks.append(f"{header}\n{s.text}")
    return "\n\n---\n\n".join(blocks).strip()
