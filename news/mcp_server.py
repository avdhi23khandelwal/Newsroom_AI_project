"""
MCP-style tool registry — REAL implementations, not mocks.

Tools:
  - web_search       : actually queries DuckDuckGo HTML and returns real results
  - fetch_url         : actually fetches a page and extracts its text
  - publish_article   : writes to the in-memory CMS store

Agents call these tools themselves, inside their own decision loop
(see agents.py). This module just executes whatever tool call an agent
decides to make and returns a JSON string result.
"""

import json
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup


# ── Tool schemas (OpenAI/Groq function-calling format) ───────────────────────
# Each agent loop passes a subset of these to the LLM so the LLM can decide,
# on its own, whether/which tool to call next.

TOOL_SCHEMAS = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent information on a topic. Returns up to 5 results with title, url, and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    "fetch_url": {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract the readable text content of a web page given its URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    "publish_article": {
        "type": "function",
        "function": {
            "name": "publish_article",
            "description": "Publish a finalized article to the CMS. Returns a CMS record with article_id and url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":   {"type": "string", "description": "Article title"},
                    "content": {"type": "string", "description": "Full article body in plain text"},
                    "topic":   {"type": "string", "description": "Topic / category tag"}
                },
                "required": ["title", "content", "topic"]
            }
        }
    }
}


def get_tool_schemas(names: list[str]) -> list[dict]:
    """Return tool schemas for the given tool names, in Groq/OpenAI format."""
    return [TOOL_SCHEMAS[n] for n in names if n in TOOL_SCHEMAS]


# ── Real tool implementations ────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NewsRoomAI/1.0; +https://newsroom.local)"
}


def _tool_web_search(query: str, max_results: int = 5) -> list[dict]:
    """Real web search via DuckDuckGo's HTML endpoint (no API key required)."""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=_HEADERS,
            timeout=10
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for result in soup.select(".result")[:max_results]:
            title_tag = result.select_one(".result__title a") or result.select_one("a.result__a")
            snippet_tag = result.select_one(".result__snippet")
            if not title_tag:
                continue
            url = title_tag.get("href", "")
            # DuckDuckGo HTML wraps redirect URLs - try to unwrap
            m = re.search(r"uddg=([^&]+)", url)
            if m:
                from urllib.parse import unquote
                url = unquote(m.group(1))
            results.append({
                "title": title_tag.get_text(strip=True),
                "url": url,
                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else ""
            })

        if not results:
            return [{"title": "", "url": "", "snippet": f"No results found for: {query}"}]
        return results

    except Exception as e:
        return [{"title": "", "url": "", "snippet": f"web_search failed: {e}"}]


def _tool_fetch_url(url: str, max_chars: int = 4000) -> str:
    """Real page fetch + text extraction."""
    if not url or not url.startswith(("http://", "https://")):
        return f"fetch_url failed: invalid URL '{url}'"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]

    except Exception as e:
        return f"fetch_url failed for {url}: {e}"


# ── Mock CMS store (publishing is the one tool we keep in-memory by design) ──

_CMS_STORE: list[dict] = []


def _tool_publish_article(title: str, content: str, topic: str) -> dict:
    article_id = f"cms-{int(time.time())}"
    record = {
        "article_id": article_id,
        "url": f"https://newsroom.local/articles/{article_id}",
        "title": title,
        "topic": topic,
        "word_count": len(content.split()),
        "published_at": datetime.utcnow().isoformat() + "Z",
        "status": "published"
    }
    _CMS_STORE.append({**record, "content": content})
    return record


def get_cms_articles():
    return _CMS_STORE


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return a JSON string result."""
    try:
        if tool_name == "web_search":
            result = _tool_web_search(tool_input.get("query", ""))
            return json.dumps(result)

        if tool_name == "fetch_url":
            result = _tool_fetch_url(tool_input.get("url", ""))
            return json.dumps({"url": tool_input.get("url", ""), "content": result})

        if tool_name == "publish_article":
            result = _tool_publish_article(
                title=tool_input.get("title", "Untitled"),
                content=tool_input.get("content", ""),
                topic=tool_input.get("topic", "general")
            )
            return json.dumps(result)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": f"Tool '{tool_name}' raised an exception: {e}"})
