# NewsRoom AI — Multi-Agent Content Pipeline

A newsroom of LLM agents that takes a topic, researches it (with real web
search), fact-checks the findings (with real web search), drafts an article,
and publishes it — with a genuine **Orchestrator agent** deciding what
happens next at every step, rather than a hardcoded sequence.

## What makes this "agentic" (not just a pipeline)

1. **Researcher and FactChecker actually call tools.** Each runs its own
   bounded loop: ask the LLM → if it requests `web_search`/`fetch_url`,
   really execute it (DuckDuckGo search + live page fetch) → feed the real
   result back → let the LLM decide whether it needs another tool call or
   is ready to answer. This repeats up to `MAX_TOOL_ITERATIONS` (4) times,
   then the agent is forced to finalize — bounded, never infinite.

2. **The Orchestrator makes real decisions.** It does not call
   Researcher → FactChecker → Writer in fixed order. At every step it asks
   the LLM, given the current state (do we have research? fact-check
   confidence? an article draft? how many retries used?), to pick ONE
   action: `research`, `fact_check`, `write`, `publish`, or `abort`. The
   LLM's choice — not an `if/else` you wrote — determines what happens next.
   For example, if FactChecker comes back with low confidence and flagged
   claims, the Orchestrator may send the topic back to the Researcher for
   another pass, or back to the Writer for a revision, before ever
   publishing.

3. **The decision loop is bounded.** `MAX_ORCHESTRATOR_STEPS` (8) caps how
   many decisions the Orchestrator can make per run. On the final step it
   is forced into a terminal action (`publish` if a draft exists, otherwise
   `abort`) so the pipeline always terminates even if the LLM keeps
   stalling. `MAX_WRITE_REVISIONS` (2) separately caps how many times the
   Writer can be sent back for a rewrite.

## Architecture

```
User Input
    |
    v
Orchestrator Agent (decision loop, bounded by MAX_ORCHESTRATOR_STEPS)
    |
    | each iteration: "given current state, what's next?" -> LLM decides
    |
    +--[action: research]----> Researcher Agent
    |                              (own tool loop: web_search, fetch_url)
    |
    +--[action: fact_check]--> FactChecker Agent
    |                              (own tool loop: web_search)
    |
    +--[action: write]-------> Writer Agent
    |                              (drafts/revises article, optional feedback)
    |
    +--[action: publish]-----> publish_article tool --> Mock CMS
    |
    +--[action: abort]-------> stop, mark run failed
```

All A2A messages, tool calls, tool results, and orchestrator decisions are
logged to SQLite (`newsroom.db`) and visible live in the dashboard.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY=gsk_...
```

(Free key at https://console.groq.com)

### 3. Run

```bash
python app.py
```

Open http://localhost:5000

## Pages

- **Dashboard** — Enter a topic, watch the live agent graph, view A2A
  message log (including tool calls/results and orchestrator decisions) +
  article preview
- **All Runs** — Table of all pipeline runs with status
- **CMS Articles** — Published articles with fact-check notes

## Files

| File | Purpose |
|------|---------|
| `app.py` | Flask web app + all HTML templates |
| `agents.py` | Orchestrator decision-loop agent + Researcher/FactChecker/Writer agents with real tool-calling loops |
| `mcp_server.py` | Real tool implementations (live `web_search` via DuckDuckGo, live `fetch_url`, `publish_article`) + tool schemas for Groq function-calling |
| `database.py` | SQLite schema + all DB helpers |
| `requirements.txt` | Python dependencies |

## Database Schema

- `runs` — one row per pipeline run
- `messages` — A2A messages between agents, including tool calls/results
  and orchestrator decisions (queryable by run_id)
- `agent_steps` — fine-grained step log per agent, including each tool call
  and each orchestrator decision
- `articles` — published articles with fact-check notes

Query messages for a run:
```sql
SELECT sender, receiver, msg_type, payload, created_at
FROM messages
WHERE run_id = '<your-run-id>'
ORDER BY id;
```

See every decision the Orchestrator made in a run:
```sql
SELECT step, result, created_at
FROM agent_steps
WHERE run_id = '<your-run-id>' AND agent = 'Orchestrator' AND step LIKE 'decision%'
ORDER BY id;
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `web_search` | Real web search (DuckDuckGo HTML, no API key needed) |
| `fetch_url` | Real page fetch + text extraction (BeautifulSoup) |
| `publish_article` | Saves to in-memory CMS store + DB |

## Loop bounds (so nothing can run forever)

| Loop | Bound | Constant |
|------|-------|----------|
| Researcher / FactChecker tool-call loop | 4 iterations, then forced final answer | `MAX_TOOL_ITERATIONS` |
| Orchestrator decision loop | 8 decisions, then forced publish/abort | `MAX_ORCHESTRATOR_STEPS` |
| Writer revision loop | 2 rewrites max | `MAX_WRITE_REVISIONS` |
| Any single LLM/tool call | 3 retries with backoff | `MAX_RETRIES` |
