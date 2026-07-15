"""
NewsRoom AI — Flask Dashboard
"""

import json
import uuid
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, redirect, url_for

from database import (
    init_db, list_runs, get_run_messages,
    get_run_steps, get_article
)
from agents import run_pipeline_background, get_pipeline_progress

app = Flask(__name__)

# ── HTML Template ─────────────────────────────────────────────────────────────

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NewsRoom AI</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
  }
  a { color: #4a9eff; text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* Layout */
  .sidebar {
    position: fixed; top: 0; left: 0;
    width: 220px; height: 100vh;
    background: #161b22;
    border-right: 1px solid #30363d;
    padding: 24px 16px;
    overflow-y: auto;
  }
  .sidebar h1 {
    font-size: 1rem; font-weight: 700; letter-spacing: .05em;
    color: #fff; margin-bottom: 24px;
    text-transform: uppercase;
  }
  .sidebar nav a {
    display: block; padding: 8px 10px; border-radius: 6px;
    color: #8b949e; font-size: .875rem; margin-bottom: 4px;
  }
  .sidebar nav a:hover, .sidebar nav a.active {
    background: #21262d; color: #e0e0e0; text-decoration: none;
  }
  .main { margin-left: 220px; padding: 32px; }

  /* Cards */
  .card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 20px;
  }
  .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 16px; color: #fff; }

  /* Form */
  .form-row { display: flex; gap: 10px; align-items: flex-start; }
  input[type="text"] {
    flex: 1; padding: 10px 14px;
    background: #0d1117; border: 1px solid #30363d;
    border-radius: 6px; color: #e0e0e0; font-size: .9rem;
  }
  input[type="text"]:focus { outline: none; border-color: #4a9eff; }
  .btn {
    padding: 10px 20px; border: none; border-radius: 6px;
    cursor: pointer; font-size: .875rem; font-weight: 600;
    transition: opacity .15s;
  }
  .btn:hover { opacity: .85; }
  .btn-primary { background: #4a9eff; color: #fff; }
  .btn-sm { padding: 5px 12px; font-size: .8rem; }
  .btn-secondary { background: #21262d; color: #e0e0e0; border: 1px solid #30363d; }

  /* Status badges */
  .badge {
    display: inline-block; padding: 2px 9px; border-radius: 12px;
    font-size: .75rem; font-weight: 600; text-transform: uppercase;
  }
  .badge-running  { background: #1f4068; color: #4a9eff; }
  .badge-completed{ background: #1a3a2a; color: #3fb950; }
  .badge-failed   { background: #3a1a1a; color: #f85149; }
  .badge-pending  { background: #2a2a1a; color: #e3b341; }

  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: .875rem; }
  th { text-align: left; padding: 8px 12px; color: #8b949e;
       border-bottom: 1px solid #30363d; font-weight: 500; }
  td { padding: 10px 12px; border-bottom: 1px solid #21262d; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }

  /* Agent graph */
  .agent-graph {
    display: flex; align-items: center; gap: 0; flex-wrap: wrap;
  }
  .agent-node {
    text-align: center; padding: 14px 18px;
    background: #21262d; border: 1px solid #30363d;
    border-radius: 8px; min-width: 110px;
  }
  .agent-node .name { font-weight: 600; font-size: .875rem; color: #fff; }
  .agent-node .role { font-size: .75rem; color: #8b949e; margin-top: 2px; }
  .agent-node.active { border-color: #4a9eff; background: #1f2d3d; }
  .agent-node.done   { border-color: #3fb950; background: #1a2d1f; }
  .agent-node.error  { border-color: #f85149; background: #2d1a1a; }
  .arrow {
    color: #30363d; font-size: 1.4rem; margin: 0 8px; user-select: none;
  }

  /* Progress log */
  .progress-log {
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
    padding: 12px; font-size: .8rem; color: #8b949e;
    max-height: 140px; overflow-y: auto; font-family: monospace;
  }
  .progress-log p { margin-bottom: 4px; }

  /* Message log */
  .msg-row { display: flex; gap: 10px; align-items: baseline; margin-bottom: 8px; }
  .msg-time { color: #484f58; font-size: .75rem; white-space: nowrap; }
  .msg-sender { font-weight: 600; font-size: .8rem; color: #4a9eff; }
  .msg-arrow  { color: #484f58; }
  .msg-receiver { font-weight: 600; font-size: .8rem; color: #3fb950; }
  .msg-type   { font-size: .75rem; color: #8b949e; margin-left: 4px; }
  .msg-payload {
    background: #0d1117; border: 1px solid #21262d; border-radius: 5px;
    padding: 8px; font-size: .75rem; color: #c9d1d9; font-family: monospace;
    margin-top: 4px; white-space: pre-wrap; word-break: break-all;
    max-height: 80px; overflow-y: auto;
  }

  /* Article preview */
  .article-title { font-size: 1.25rem; font-weight: 700; color: #fff; margin-bottom: 12px; }
  .article-body  { font-size: .9rem; line-height: 1.7; color: #c9d1d9; white-space: pre-wrap; }
  .fact-check-box {
    background: #1a2d1f; border-left: 3px solid #3fb950;
    padding: 12px 16px; border-radius: 0 6px 6px 0; margin-top: 16px;
  }
  .fact-check-box h3 { font-size: .8rem; color: #3fb950; margin-bottom: 6px; }
  .fact-check-box p  { font-size: .8rem; color: #8b949e; }

  /* Steps list */
  .step-item { display: flex; gap: 10px; align-items: baseline; margin-bottom: 6px; }
  .step-agent { font-size: .75rem; font-weight: 600; color: #4a9eff; min-width: 90px; }
  .step-name  { font-size: .8rem; color: #c9d1d9; }
  .step-time  { font-size: .7rem; color: #484f58; margin-left: auto; }
  .step-error { font-size: .75rem; color: #f85149; }

  /* Grid */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

  /* Empty state */
  .empty { color: #484f58; font-size: .875rem; text-align: center; padding: 24px 0; }

  /* Spinner */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid #30363d; border-top-color: #4a9eff;
    border-radius: 50%; animation: spin .8s linear infinite;
    vertical-align: middle; margin-right: 6px;
  }
</style>
</head>
<body>
<div class="sidebar">
  <h1>NewsRoom AI</h1>
  <nav>
    <a href="/" class="{{ 'active' if page=='home' else '' }}">Dashboard</a>
    <a href="/runs" class="{{ 'active' if page=='runs' else '' }}">All Runs</a>
    <a href="/cms" class="{{ 'active' if page=='cms' else '' }}">CMS Articles</a>
  </nav>
</div>
<div class="main">
{% block content %}{% endblock %}
</div>
<script>
function autoRefresh(ms) {
  setTimeout(() => location.reload(), ms);
}
</script>
{% block scripts %}{% endblock %}
</body>
</html>
"""

INDEX_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="card">
  <h2>Start Pipeline</h2>
  <form method="POST" action="/start">
    <div class="form-row">
      <input type="text" name="topic" placeholder="Enter a news topic..." required
             value="{{ topic or '' }}">
      <button type="submit" class="btn btn-primary">Run Pipeline</button>
    </div>
  </form>
</div>

{% if run_id %}
<!-- Agent Graph -->
<div class="card">
  <h2>Agent Pipeline</h2>
  <div class="agent-graph">
    {% for node in graph_nodes %}
      <div class="agent-node {{ node.state }}">
        <div class="name">{{ node.name }}</div>
        <div class="role">{{ node.role }}</div>
      </div>
      {% if not loop.last %}<div class="arrow">&#8594;</div>{% endif %}
    {% endfor %}
  </div>
  <div style="margin-top:16px;">
    <div class="progress-log" id="prog-log">
      {% for line in progress %}
      <p>{{ line }}</p>
      {% endfor %}
      {% if not progress %}
      <p>Waiting for progress...</p>
      {% endif %}
    </div>
  </div>
</div>

<div class="grid-2">
  <!-- A2A Message Log -->
  <div class="card">
    <h2>A2A Message Log</h2>
    {% if messages %}
      {% for m in messages %}
      <div class="msg-row">
        <span class="msg-time">{{ m.created_at[11:19] }}</span>
        <span class="msg-sender">{{ m.sender }}</span>
        <span class="msg-arrow">&#8594;</span>
        <span class="msg-receiver">{{ m.receiver }}</span>
        <span class="msg-type">[{{ m.msg_type }}]</span>
      </div>
      <div class="msg-payload">{{ m.payload[:300] }}{% if m.payload|length > 300 %}...{% endif %}</div>
      {% endfor %}
    {% else %}
      <div class="empty">No messages yet</div>
    {% endif %}
  </div>

  <!-- Agent Steps -->
  <div class="card">
    <h2>Agent Steps</h2>
    {% if steps %}
      {% for s in steps %}
      <div class="step-item">
        <span class="step-agent">{{ s.agent }}</span>
        <span class="step-name">{{ s.step }}</span>
        <span class="step-time">{{ s.created_at[11:19] }}</span>
      </div>
      {% if s.error %}<div class="step-error">  Error: {{ s.error[:120] }}</div>{% endif %}
      {% endfor %}
    {% else %}
      <div class="empty">No steps yet</div>
    {% endif %}
  </div>
</div>

<!-- Article Preview -->
{% if article %}
<div class="card">
  <h2>Article Preview</h2>
  <div class="article-title">{{ article.topic }}</div>
  <div class="article-body">{{ article.content }}</div>
  {% if article.fact_check_notes %}
  <div class="fact-check-box">
    <h3>Fact-Check Notes</h3>
    <p>{{ article.fact_check_notes }}</p>
  </div>
  {% endif %}
</div>
{% endif %}

{% if run_status == 'running' %}
<script>autoRefresh(3000);</script>
{% endif %}
{% endif %}
""").replace("{% block scripts %}{% endblock %}", "")

RUNS_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="card">
  <h2>All Pipeline Runs</h2>
  {% if runs %}
  <table>
    <thead>
      <tr>
        <th>Run ID</th>
        <th>Topic</th>
        <th>Status</th>
        <th>Messages</th>
        <th>Started</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      {% for r in runs %}
      <tr>
        <td style="font-family:monospace;font-size:.75rem;color:#484f58">{{ r.run_id[:16] }}...</td>
        <td>{{ r.topic }}</td>
        <td><span class="badge badge-{{ r.status }}">{{ r.status }}</span></td>
        <td>{{ r.msg_count }}</td>
        <td style="font-size:.8rem;color:#8b949e">{{ r.created_at[:19].replace('T',' ') }}</td>
        <td><a href="/?run_id={{ r.run_id }}&topic={{ r.topic }}" class="btn btn-sm btn-secondary">View</a></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty">No runs yet. Start a pipeline from the Dashboard.</div>
  {% endif %}
</div>
""").replace("{% block scripts %}{% endblock %}", "")

CMS_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="card">
  <h2>CMS — Published Articles</h2>
  {% if articles %}
    {% for a in articles %}
    <div style="border-bottom:1px solid #21262d; padding-bottom:20px; margin-bottom:20px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
        <div class="article-title" style="font-size:1rem;">{{ a.title }}</div>
        <span class="badge badge-completed">published</span>
      </div>
      <div style="font-size:.75rem;color:#484f58;margin-bottom:10px;">
        {{ a.topic }} &bull; {{ a.word_count }} words &bull; {{ a.published_at[:19].replace('T',' ') }} UTC
      </div>
      <div class="article-body" style="max-height:200px;overflow-y:auto;">{{ a.content }}</div>
      {% if a.fact_check_notes %}
      <div class="fact-check-box" style="margin-top:10px;">
        <h3>Fact-Check Notes</h3>
        <p>{{ a.fact_check_notes }}</p>
      </div>
      {% endif %}
    </div>
    {% endfor %}
  {% else %}
  <div class="empty">No articles published yet.</div>
  {% endif %}
</div>
""").replace("{% block scripts %}{% endblock %}", "")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    run_id = request.args.get("run_id")
    topic  = request.args.get("topic", "")

    graph_nodes = [
        {"name": "Orchestrator", "role": "Task Router",    "state": ""},
        {"name": "Researcher",   "role": "Web Search",     "state": ""},
        {"name": "FactChecker",  "role": "Verification",   "state": ""},
        {"name": "Writer",       "role": "Article Draft",  "state": ""},
        {"name": "CMS",          "role": "Publish",        "state": ""},
    ]

    messages   = []
    steps      = []
    article    = None
    run_status = None
    progress   = []

    if run_id:
        from database import get_connection
        conn = get_connection()
        row  = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
        conn.close()
        run_status = dict(row)["status"] if row else "unknown"
        messages   = get_run_messages(run_id)
        steps      = get_run_steps(run_id)
        article    = get_article(run_id)
        progress   = get_pipeline_progress(run_id)

        # Color graph nodes based on steps
        done_agents = {s["agent"] for s in steps if s["step"] == "complete"}
        error_agents = {s["agent"] for s in steps if s["step"] == "error"}
        active_agents = {s["agent"] for s in steps if s["step"] == "start"} - done_agents

        for node in graph_nodes:
            n = node["name"]
            if n in error_agents:
                node["state"] = "error"
            elif n in done_agents or (n == "CMS" and run_status == "completed"):
                node["state"] = "done"
            elif n in active_agents:
                node["state"] = "active"

    return render_template_string(
        INDEX_HTML,
        page="home", run_id=run_id, topic=topic,
        graph_nodes=graph_nodes, messages=messages,
        steps=steps, article=article,
        run_status=run_status, progress=progress
    )


@app.route("/start", methods=["POST"])
def start():
    topic  = request.form.get("topic", "").strip()
    if not topic:
        return redirect(url_for("index"))
    run_id = str(uuid.uuid4())
    run_pipeline_background(run_id, topic)
    return redirect(url_for("index", run_id=run_id, topic=topic))


@app.route("/runs")
def runs_page():
    runs = list_runs()
    return render_template_string(RUNS_HTML, page="runs", runs=runs)


@app.route("/cms")
def cms_page():
    from database import get_connection
    from mcp_server import get_cms_articles
    conn = get_connection()
    db_articles = conn.execute(
        "SELECT topic, content, fact_check_notes, published_at FROM articles ORDER BY id DESC"
    ).fetchall()
    conn.close()

    cms_articles = get_cms_articles()

    display = []
    for i, row in enumerate(db_articles):
        d = dict(row)
        cms_idx = len(cms_articles) - 1 - i
        if 0 <= cms_idx < len(cms_articles):
            d["title"]      = cms_articles[cms_idx].get("title", d["topic"])
            d["article_id"] = cms_articles[cms_idx].get("article_id", "")
            d["word_count"] = cms_articles[cms_idx].get("word_count", 0)
        else:
            d["title"]      = d["topic"]
            d["article_id"] = ""
            d["word_count"] = len(d["content"].split())
        display.append(d)

    return render_template_string(CMS_HTML, page="cms", articles=display)


@app.route("/api/status/<run_id>")
def api_status(run_id):
    from database import get_connection
    conn = get_connection()
    row  = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    status   = dict(row)["status"] if row else "unknown"
    progress = get_pipeline_progress(run_id)
    return jsonify({"run_id": run_id, "status": status, "progress": progress})


@app.route("/api/messages/<run_id>")
def api_messages(run_id):
    return jsonify(get_run_messages(run_id))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("NewsRoom AI starting at http://localhost:5000")
    app.run(debug=False, port=5000, use_reloader=False)
