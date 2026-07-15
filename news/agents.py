"""
Multi-agent pipeline using Groq (free, no SDK — raw HTTP via requests):

  Orchestrator (decision-loop agent) routes between:
      Researcher  <-- uses web_search / fetch_url tools in its own loop
      FactChecker <-- uses web_search tool in its own loop to verify claims
      Writer      <-- drafts the article
      publish_article tool

  The Orchestrator is a real agent: at every step it asks the LLM
  "given everything so far, what should happen next?" and the LLM's
  decision (not a hardcoded if/else) determines whether to research more,
  re-fact-check, revise the article, or publish.

  The loop is BOUNDED (MAX_ORCHESTRATOR_STEPS) so a confused agent can
  never spin forever — it is forced to a decision once the budget runs out.

Set env var before running:
    export GROQ_API_KEY=gsk_...
    python app.py
"""

import os
import json
import re
import time
import threading

import requests
from dotenv import load_dotenv

load_dotenv()


from database import (
    log_message, log_step, create_run,
    update_run_status, save_article
)
from mcp_server import dispatch_tool, get_tool_schemas

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"



MAX_TOKENS   = 1500
MAX_RETRIES  = 3
RETRY_DELAY  = 2   # seconds

# Bounds that keep every loop in this file finite.
MAX_TOOL_ITERATIONS       = 4   # per-agent tool-call loop (Researcher / FactChecker)
MAX_ORCHESTRATOR_STEPS    = 8   # top-level decision loop
MAX_WRITE_REVISIONS       = 2   # how many times Orchestrator may send article back to Writer


# ── Core API call ─────────────────────────────────────────────────────────────

def _call_groq(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """
    Call the Groq chat-completions API. Returns the raw `message` dict from
    the response (may contain `content` and/or `tool_calls`).
    Retries automatically on rate-limit / server errors.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at https://console.groq.com and run: "
            "export GROQ_API_KEY=gsk_..."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    body = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.4,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(GROQ_API_URL, json=body, headers=headers, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]
            if resp.status_code in (429, 500, 502, 503) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                continue
            resp.raise_for_status()
        except requests.Timeout as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(f"Groq API failed after {MAX_RETRIES} retries: {last_err}")


def _parse_json_safe(text: str) -> dict:
    """Parse JSON from model output, handling common formatting issues."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def fix_newlines(m):
        return m.group(0).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    text = re.sub(r'"(?:[^"\\]|\\.)*"', fix_newlines, text, flags=re.DOTALL)

    return json.loads(text)


# ── Tool-calling agent loop (shared by Researcher & FactChecker) ─────────────

def _run_tool_agent(run_id: str, agent_name: str, system: str, user_msg: str,
                     tool_names: list[str], max_iterations: int = MAX_TOOL_ITERATIONS) -> dict:
    """
    Generic bounded agent loop:
      1. Ask the LLM, giving it tools.
      2. If it asks for a tool call -> execute it for real, feed the result back, loop.
      3. If it returns a final answer -> parse and return it.
      4. If max_iterations is hit -> force a final, tool-less call to wrap up.

    This is the actual "agent" part: the LLM decides each iteration whether
    it needs another tool call or has enough information to stop.
    """
    tools = get_tool_schemas(tool_names)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    for iteration in range(1, max_iterations + 1):
        message = _call_groq(messages, tools=tools)
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            # Agent decided it's done.
            log_step(run_id, agent_name, f"final_answer_iter_{iteration}")
            return _parse_json_safe(message.get("content", "{}"))

        # Agent wants to use one or more tools — execute them for real.
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls
        })

        for call in tool_calls:
            fn_name = call["function"]["name"]
            try:
                fn_args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            log_step(run_id, agent_name, f"tool_call_{fn_name}",
                     result=json.dumps(fn_args))
            log_message(run_id, agent_name, "MCP", "tool_call",
                        {"tool": fn_name, "input": fn_args})

            tool_result = dispatch_tool(fn_name, fn_args)

            log_message(run_id, "MCP", agent_name, "tool_result",
                        {"tool": fn_name, "result": tool_result[:500]})

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": tool_result
            })

    # Iteration budget exhausted — force one last tool-less call to finalize.
    log_step(run_id, agent_name, "force_finalize",
             result=f"Hit max_iterations={max_iterations}, forcing final answer")
    messages.append({
        "role": "user",
        "content": "You are out of tool calls. Based on everything gathered so far, "
                    "give your final answer now as the required JSON object only."
    })
    message = _call_groq(messages, tools=None)
    return _parse_json_safe(message.get("content", "{}"))


# ── Agents ────────────────────────────────────────────────────────────────────

class ResearcherAgent:
    NAME = "Researcher"

    def run(self, run_id: str, topic: str) -> dict:
        log_step(run_id, self.NAME, "start", f"Researching: {topic}")
        log_message(run_id, "Orchestrator", self.NAME, "task",
                    {"task": "research", "topic": topic})

        system = (
            "You are a professional news researcher. You have access to a `web_search` "
            "tool and a `fetch_url` tool — USE THEM. Do not rely only on your training "
            "knowledge; search for current information and fetch promising pages to confirm "
            "details before answering. Call web_search at least once, and fetch_url on at "
            "least one promising result, before giving your final answer.\n\n"
            "When you are confident you have enough verified information, respond with "
            "ONLY a valid JSON object (no markdown, no tool call) with these exact keys:\n"
            "  'summary': a 2-3 sentence overview of the topic\n"
            "  'key_facts': a list of 5-6 specific facts as strings\n"
            "  'sources': a list of the actual URLs you fetched or found via search\n"
        )
        user_msg = f"Research this news topic thoroughly using your tools: {topic}"

        try:
            data = _run_tool_agent(run_id, self.NAME, system, user_msg,
                                    tool_names=["web_search", "fetch_url"])
        except Exception as e:
            log_step(run_id, self.NAME, "error", error=str(e))
            data = {
                "summary": f"Research on '{topic}' could not be completed automatically.",
                "key_facts": [f"Topic under investigation: {topic}"],
                "sources": []
            }

        log_step(run_id, self.NAME, "complete", result=json.dumps(data)[:500])
        log_message(run_id, self.NAME, "Orchestrator", "research_result", data)
        return data


class FactCheckerAgent:
    NAME = "FactChecker"

    def run(self, run_id: str, topic: str, research: dict) -> dict:
        log_step(run_id, self.NAME, "start", f"Fact-checking: {topic}")
        log_message(run_id, "Orchestrator", self.NAME, "task",
                    {"task": "fact_check", "topic": topic, "research": research})

        key_facts = research.get("key_facts", [])
        facts_str = "\n".join(f"- {f}" for f in key_facts)

        system = (
            "You are a meticulous fact-checker. You have access to a `web_search` tool — "
            "USE IT to independently verify claims rather than judging them from memory alone. "
            "Search for at least one claim you are unsure about before finalizing.\n\n"
            "When done, respond with ONLY a valid JSON object (no markdown, no tool call) "
            "with these exact keys:\n"
            "  'verified_facts': list of claims that appear accurate (reworded for clarity)\n"
            "  'flagged': list of claims that seem uncertain, exaggerated, or unverifiable\n"
            "  'confidence': one of 'high', 'medium', or 'low'\n"
            "  'notes': 1-2 sentences summarizing your fact-check findings\n"
        )
        user_msg = (
            f"Fact-check these claims about '{topic}' using web_search where helpful:\n\n"
            f"{facts_str}"
        )

        try:
            data = _run_tool_agent(run_id, self.NAME, system, user_msg,
                                    tool_names=["web_search"])
        except Exception as e:
            log_step(run_id, self.NAME, "error", error=str(e))
            data = {
                "verified_facts": key_facts,
                "flagged": [],
                "confidence": "medium",
                "notes": "Automated fact-check could not complete all verifications."
            }

        log_step(run_id, self.NAME, "complete", result=json.dumps(data)[:500])
        log_message(run_id, self.NAME, "Orchestrator", "factcheck_result", data)
        return data


class WriterAgent:
    NAME = "Writer"

    def run(self, run_id: str, topic: str, research: dict, fact_check: dict,
            revision_feedback: str | None = None) -> dict:
        log_step(run_id, self.NAME, "start", f"Writing article: {topic}")
        log_message(run_id, "Orchestrator", self.NAME, "task",
                    {"task": "write", "topic": topic, "revision_feedback": revision_feedback})

        verified  = fact_check.get("verified_facts", research.get("key_facts", []))
        summary   = research.get("summary", "")
        fc_notes  = fact_check.get("notes", "")
        facts_str = "\n".join(f"- {f}" for f in verified)

        system = (
            "You are a professional news journalist. "
            "Write a concise, factual news article (350-500 words) based on the provided research. "
            "Use this structure: strong headline, lead paragraph (who/what/when/where/why), "
            "3-4 body paragraphs with details, brief conclusion.\n"
            "Return ONLY a valid JSON object with these exact keys:\n"
            "  'title': the article headline as a string\n"
            "  'body': the full article text with paragraphs separated by newlines\n"
            "No markdown, no code fences, no extra text. Just the JSON object."
        )
        content = (
            f"Write a news article about: {topic}\n\n"
            f"Research summary: {summary}\n\n"
            f"Verified facts to include:\n{facts_str}\n\n"
            f"Fact-check notes: {fc_notes}"
        )
        if revision_feedback:
            content += f"\n\nThe Orchestrator reviewed a previous draft and requested this revision:\n{revision_feedback}"

        try:
            message = _call_groq(
                [{"role": "system", "content": system},
                 {"role": "user", "content": content}],
                tools=None
            )
            data = _parse_json_safe(message.get("content", "{}"))
        except Exception as e:
            log_step(run_id, self.NAME, "error", error=str(e))
            data = {
                "title": f"Report: {topic}",
                "body": f"{summary}\n\n" + "\n".join(verified)
            }

        log_step(run_id, self.NAME, "complete",
                 result=json.dumps({"title": data.get("title")}))
        log_message(run_id, self.NAME, "Orchestrator", "article_draft", data)
        return data


# ── Orchestrator: a real decision-making agent, not a fixed sequence ────────

class OrchestratorAgent:
    """
    At every step, the Orchestrator asks the LLM:
        "Given everything that has happened so far, what should happen next?"
    The LLM picks one action from a fixed menu, and the Orchestrator executes
    it (by calling the relevant agent or tool). This repeats until the LLM
    decides to publish, or until MAX_ORCHESTRATOR_STEPS is reached — at which
    point the Orchestrator forces a publish-or-abort decision so the loop
    can never run forever.

    Possible actions the LLM can choose:
      - "research"        : run/re-run the Researcher (e.g. not enough facts yet)
      - "fact_check"       : run/re-run the FactChecker on current research
      - "write"            : run/re-run the Writer (optionally with revision feedback)
      - "publish"          : publish the current article draft
      - "abort"            : give up (e.g. fact-check confidence too low after retries)
    """

    NAME = "Orchestrator"

    DECISION_SYSTEM = (
        "You are the Orchestrator agent for an automated newsroom. You do not write or "
        "research anything yourself — you only decide what should happen next, by picking "
        "ONE action from this list:\n"
        "  research    - (re)run the Researcher agent to gather more/better facts\n"
        "  fact_check   - (re)run the FactChecker agent on the current research\n"
        "  write        - (re)run the Writer agent to draft or revise the article\n"
        "  publish      - publish the current article draft as-is\n"
        "  abort        - stop the pipeline (e.g. confidence is low and retries are exhausted)\n\n"
        "Rules of thumb:\n"
        "- If there is no research yet, choose 'research'.\n"
        "- If research exists but hasn't been fact-checked, choose 'fact_check'.\n"
        "- If fact_check confidence is 'low' and flagged claims exist, and you have research "
        "  retries left, choose 'research' again with revised guidance, otherwise 'fact_check' again.\n"
        "- If research + fact_check look solid but there is no article draft yet, choose 'write'.\n"
        "- If an article draft exists and looks complete/accurate, choose 'publish'.\n"
        "- If an article draft exists but contradicts verified facts or is too short, choose "
        "  'write' again with feedback explaining what to fix (only if revisions remain).\n"
        "- Never repeat the exact same action more than necessary; move the pipeline forward.\n\n"
        "Respond with ONLY a JSON object: "
        "{\"action\": \"research|fact_check|write|publish|abort\", "
        "\"feedback\": \"short guidance for the next agent, or empty string\", "
        "\"reason\": \"one sentence explaining your decision\"}"
    )

    def __init__(self):
        self.researcher   = ResearcherAgent()
        self.fact_checker = FactCheckerAgent()
        self.writer       = WriterAgent()

    def _decide(self, run_id: str, state: dict) -> dict:
        """Ask the LLM what to do next, given the current pipeline state."""
        state_summary = {
            "topic": state["topic"],
            "has_research": bool(state["research"]),
            "has_fact_check": bool(state["fact_check"]),
            "fact_check_confidence": state["fact_check"].get("confidence") if state["fact_check"] else None,
            "flagged_claims": state["fact_check"].get("flagged") if state["fact_check"] else [],
            "has_article": bool(state["article"]),
            "research_attempts": state["research_attempts"],
            "fact_check_attempts": state["fact_check_attempts"],
            "write_attempts": state["write_attempts"],
            "steps_taken_so_far": state["step_count"],
            "steps_remaining_budget": MAX_ORCHESTRATOR_STEPS - state["step_count"],
        }

        message = _call_groq(
            [
                {"role": "system", "content": self.DECISION_SYSTEM},
                {"role": "user", "content": f"Current pipeline state:\n{json.dumps(state_summary, indent=2)}\n\nWhat should happen next?"}
            ],
            tools=None
        )
        try:
            decision = _parse_json_safe(message.get("content", "{}"))
        except Exception:
            decision = {"action": "publish" if state["article"] else "abort",
                        "feedback": "", "reason": "fallback after decision parse failure"}

        log_step(run_id, self.NAME, f"decision_{state['step_count']}",
                 result=json.dumps(decision))
        log_message(run_id, self.NAME, self.NAME, "decision", decision)
        return decision

    def run(self, run_id: str, topic: str, progress_cb=None) -> dict:
        def _progress(msg):
            if progress_cb:
                progress_cb(msg)

        create_run(run_id, topic)
        log_message(run_id, "User", self.NAME, "request", {"topic": topic})
        log_step(run_id, self.NAME, "start", f"Pipeline started for: {topic}")

        state = {
            "topic": topic,
            "research": None,
            "fact_check": None,
            "article": None,
            "research_attempts": 0,
            "fact_check_attempts": 0,
            "write_attempts": 0,
            "step_count": 0,
        }

        try:
            while state["step_count"] < MAX_ORCHESTRATOR_STEPS:
                state["step_count"] += 1

                # Force a terminal decision once the budget is almost gone,
                # so the loop is guaranteed to end even if the LLM keeps stalling.
                if state["step_count"] == MAX_ORCHESTRATOR_STEPS:
                    decision = {
                        "action": "publish" if state["article"] else "abort",
                        "feedback": "",
                        "reason": "Orchestrator step budget exhausted; forcing terminal action."
                    }
                    log_step(run_id, self.NAME, "forced_terminal_decision",
                             result=json.dumps(decision))
                else:
                    decision = self._decide(run_id, state)

                action   = decision.get("action", "abort")
                feedback = decision.get("feedback", "")

                if action == "research":
                    state["research_attempts"] += 1
                    _progress(f"Orchestrator → Researcher (attempt {state['research_attempts']}): {decision.get('reason','')}")
                    state["research"] = self._retry(
                        lambda: self.researcher.run(run_id, topic), run_id, "research"
                    )

                elif action == "fact_check":
                    if not state["research"]:
                        # Can't fact-check nothing — force research first.
                        continue
                    state["fact_check_attempts"] += 1
                    _progress(f"Orchestrator → FactChecker (attempt {state['fact_check_attempts']}): {decision.get('reason','')}")
                    state["fact_check"] = self._retry(
                        lambda: self.fact_checker.run(run_id, topic, state["research"]),
                        run_id, "fact_check"
                    )

                elif action == "write":
                    if not state["research"]:
                        continue
                    if state["write_attempts"] >= MAX_WRITE_REVISIONS:
                        # Revision budget spent — stop asking for rewrites, move to publish/abort.
                        continue
                    state["write_attempts"] += 1
                    _progress(f"Orchestrator → Writer (attempt {state['write_attempts']}): {decision.get('reason','')}")
                    state["article"] = self._retry(
                        lambda: self.writer.run(
                            run_id, topic, state["research"],
                            state["fact_check"] or {}, revision_feedback=feedback or None
                        ),
                        run_id, "write"
                    )

                elif action == "publish":
                    if not state["article"]:
                        # Nothing to publish yet — let the loop continue deciding.
                        continue
                    _progress("Orchestrator → Publishing to CMS...")
                    title = state["article"].get("title", f"Report: {topic}")
                    body  = state["article"].get("body", "")
                    pub_result = json.loads(dispatch_tool(
                        "publish_article",
                        {"title": title, "content": body, "topic": topic}
                    ))
                    log_message(run_id, self.NAME, "CMS", "publish", pub_result)

                    save_article(run_id, topic, body,
                                (state["fact_check"] or {}).get("notes", ""))
                    update_run_status(run_id, "completed")
                    log_step(run_id, self.NAME, "complete",
                             result=pub_result.get("article_id"))
                    _progress("Done.")

                    return {
                        "status": "success",
                        "run_id": run_id,
                        "article": {"title": title, "body": body},
                        "fact_check": state["fact_check"],
                        "cms": pub_result
                    }

                else:  # "abort" or anything unrecognized
                    update_run_status(run_id, "failed")
                    reason = decision.get("reason", "Orchestrator chose to abort.")
                    log_step(run_id, self.NAME, "aborted", error=reason)
                    _progress(f"Orchestrator aborted: {reason}")
                    return {"status": "error", "run_id": run_id, "error": reason}

            # Loop exited without ever returning — shouldn't normally happen
            # given the forced terminal decision above, but guard anyway.
            update_run_status(run_id, "failed")
            return {"status": "error", "run_id": run_id,
                    "error": "Orchestrator exhausted its step budget without finishing."}

        except Exception as e:
            update_run_status(run_id, "failed")
            log_step(run_id, self.NAME, "error", error=str(e))
            _progress(f"Pipeline failed: {e}")
            return {"status": "error", "run_id": run_id, "error": str(e)}

    def _retry(self, fn, run_id, step_name):
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as e:
                last_err = e
                log_step(run_id, self.NAME,
                         f"retry_{step_name}_{attempt}", error=str(e))
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
        raise RuntimeError(
            f"{step_name} failed after {MAX_RETRIES} retries: {last_err}"
        )


# Backwards-compatible alias — app.py / older code may import `Orchestrator`.
Orchestrator = OrchestratorAgent


# ── Background runner ─────────────────────────────────────────────────────────

_pipeline_status: dict[str, list[str]] = {}


def run_pipeline_background(run_id: str, topic: str):
    """Run the full pipeline in a background thread."""
    _pipeline_status[run_id] = []

    def _progress(msg):
        _pipeline_status[run_id].append(msg)

    def _thread():
        orch = OrchestratorAgent()
        orch.run(run_id, topic, progress_cb=_progress)

    t = threading.Thread(target=_thread, daemon=True)
    t.start()


def get_pipeline_progress(run_id: str) -> list[str]:
    return _pipeline_status.get(run_id, [])
