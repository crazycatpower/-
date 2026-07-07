# -*- coding: utf-8 -*-
"""
Agentic tool-use loop for Ollama models.

Tools exposed to the model:
  - search_regulations(query, top_k)  → queries regulations / regulations_kb tables
  - query_audit_history(days, limit)  → queries audit_records table

Usage:
    from tools_chat import chat_with_tools, TOOLS
    import openai
    client = openai.OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
    result = chat_with_tools(client, "mistral:latest", messages)
"""

import json
import time

import openai

# ── Tool schema (OpenAI function-calling format) ──────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": (
                "搜尋台灣職業安全衛生法規資料庫，回傳相關條文與說明。"
                "當需要引用具體法條、規則條號或法規內容時呼叫此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜尋關鍵字，例如：高架作業、安全帽、防墜措施、電氣安全",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "回傳筆數，預設 4，最多 8",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_audit_history",
            "description": (
                "查詢近期現場稽核記錄，回傳統計數量與摘要。"
                "當需要了解歷史違規趨勢或最近稽核狀況時呼叫。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "查詢天數範圍，預設 30",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "回傳最近幾筆記錄，預設 5",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Tool executors ─────────────────────────────────────────────────────────────

def _exec_search_regulations(query: str, top_k: int = 4) -> str:
    top_k = min(max(1, int(top_k)), 8)
    try:
        from knowledge_base import retrieve_regulations_from_db, format_snippets
        snips = retrieve_regulations_from_db(query, top_k=top_k)
        if not snips:
            return f"資料庫中找不到與「{query}」相關的法規條文。"
        return format_snippets(snips)
    except Exception as e:
        return f"法規查詢失敗：{e}"


def _exec_query_audit_history(days: int = 30, limit: int = 5) -> str:
    days = max(1, int(days))
    limit = min(max(1, int(limit)), 20)
    try:
        from db import db_conn
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS total,
                           COALESCE(SUM(detection_count), 0) AS total_det
                    FROM audit_records
                    WHERE SUBSTR(created_at, 1, 10) >=
                          TO_CHAR(NOW() - INTERVAL '1 day' * %s, 'YYYY-MM-DD')
                    """,
                    (days,),
                )
                stats = cur.fetchone() or {}

                cur.execute(
                    """
                    SELECT created_at, summary, detection_count, analysis_mode
                    FROM audit_records
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                recent = cur.fetchall()

        lines = [
            f"近 {days} 天稽核：共 {stats.get('total', 0)} 筆，"
            f"偵測物件 {stats.get('total_det', 0)} 個。"
        ]
        if recent:
            lines.append("最近稽核記錄：")
            for r in recent:
                summary_short = (r.get("summary") or "")[:80]
                lines.append(
                    f"  [{str(r.get('created_at', ''))[:10]}]"
                    f" {r.get('analysis_mode', '')} |"
                    f" 偵測 {r.get('detection_count', 0)} 個 |"
                    f" {summary_short}"
                )
        return "\n".join(lines)
    except Exception as e:
        return f"稽核記錄查詢失敗：{e}"


def execute_tool(name: str, arguments: dict) -> str:
    if name == "search_regulations":
        return _exec_search_regulations(
            query=arguments.get("query", ""),
            top_k=arguments.get("top_k", 4),
        )
    if name == "query_audit_history":
        return _exec_query_audit_history(
            days=arguments.get("days", 30),
            limit=arguments.get("limit", 5),
        )
    return f"未知工具：{name}"


# ── Agentic loop ───────────────────────────────────────────────────────────────

def chat_with_tools(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    system_prompt: str | None = None,
    max_turns: int = 5,
    temperature: float = 0.1,
    timeout: int = 120,
) -> dict:
    """
    Send messages to model; if it requests a tool call, execute it and
    continue the conversation until a final text answer is produced.

    Returns:
        {
            "output": str,                  # final answer
            "tool_calls_made": list[dict],  # each call's name, args, result preview
            "turns_used": int,
            "total_time": float,
            "error": str | None,
        }
    """
    result = {
        "output": "",
        "tool_calls_made": [],
        "turns_used": 0,
        "total_time": None,
        "error": None,
    }

    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(messages)

    t_start = time.perf_counter()

    for turn in range(max_turns):
        result["turns_used"] = turn + 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
                timeout=timeout,
                stream=False,
            )
        except Exception as e:
            result["error"] = str(e)
            break

        choice = resp.choices[0]
        msg = choice.message

        # ── No tool calls → final answer ──
        if not msg.tool_calls:
            result["output"] = msg.content or ""
            break

        # ── Append assistant turn (with tool_calls) ──
        msgs.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # ── Execute each tool and append results ──
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception as _json_err:
                print(f"[tools] {tc.function.name} 參數解析失敗：{_json_err}｜原始：{tc.function.arguments[:200]}")
                args = {}

            tool_result = execute_tool(tc.function.name, args)

            result["tool_calls_made"].append({
                "tool": tc.function.name,
                "args": args,
                "result_preview": tool_result[:300],
            })

            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    result["total_time"] = round(time.perf_counter() - t_start, 3)
    return result
