"""
CLASP Real-Time Telemetry Dashboard
3D UMAP projection of agent embeddings + trajectory progress + live stats.
"""
from __future__ import annotations
import json
import threading
import time
from collections import deque

import numpy as np
import plotly.graph_objects as go
import redis
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output

from configs.settings import REDIS_HOST, REDIS_PORT, DASH_HOST, DASH_PORT, DEFAULT_AGENTS

# ── Telemetry state (written by orchestrator, read by Dash) ─────────────────
# Stored in Redis key "clasp:telemetry" as a JSON list of event dicts

app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "#0d1117", "minHeight": "100vh", "fontFamily": "monospace"},
    children=[
        html.H1(
            "CLASP — Cosmos Learning Agent Safety Protocol",
            style={"color": "#58a6ff", "padding": "20px", "margin": 0},
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "padding": "0 20px"},
            children=[
                # Left: UMAP 3D scatter
                html.Div([
                    html.H3("Latent Space Manifold (UMAP 3D)", style={"color": "#8b949e"}),
                    dcc.Graph(id="umap-graph", style={"height": "500px"}),
                ]),
                # Right: Agent stats
                html.Div([
                    html.H3("Agent Live Stats", style={"color": "#8b949e"}),
                    dcc.Graph(id="agent-bar", style={"height": "240px"}),
                    html.H3("Trajectory Progress", style={"color": "#8b949e"}),
                    dcc.Graph(id="trajectory-line", style={"height": "240px"}),
                ]),
            ],
        ),
        html.Div(
            style={"padding": "20px"},
            children=[
                html.H3("Event Log", style={"color": "#8b949e"}),
                html.Div(id="event-log", style={
                    "backgroundColor": "#161b22",
                    "color": "#c9d1d9",
                    "padding": "12px",
                    "height": "200px",
                    "overflowY": "scroll",
                    "fontSize": "12px",
                    "fontFamily": "monospace",
                }),
            ],
        ),
        dcc.Interval(id="refresh", interval=2000, n_intervals=0),
    ],
)


def _get_events() -> list[dict]:
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        raw = r.get("clasp:telemetry")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return []


@callback(
    Output("umap-graph", "figure"),
    Output("agent-bar", "figure"),
    Output("trajectory-line", "figure"),
    Output("event-log", "children"),
    Input("refresh", "n_intervals"),
)
def update_dashboard(_):
    events = _get_events()

    dark = {"paper_bgcolor": "#161b22", "plot_bgcolor": "#0d1117",
            "font": {"color": "#c9d1d9"}, "margin": dict(l=40, r=20, t=30, b=30)}

    # ── UMAP 3D ─────────────────────────────────────────────────────────────
    umap_fig = go.Figure(layout=go.Layout(**dark))
    if events:
        embeddings = [e.get("embedding_snippet", []) for e in events if e.get("embedding_snippet")]
        if len(embeddings) >= 5:
            try:
                import umap as umap_lib
                X = np.array(embeddings)
                n = min(len(X), 3)
                reducer = umap_lib.UMAP(n_components=3, n_neighbors=n, random_state=42)
                Y = reducer.fit_transform(X)
                decisions = [e.get("consensus", False) for e in events if e.get("embedding_snippet")]
                colors = ["#3fb950" if d else "#f85149" for d in decisions]
                umap_fig.add_trace(go.Scatter3d(
                    x=Y[:, 0], y=Y[:, 1], z=Y[:, 2],
                    mode="markers",
                    marker=dict(size=5, color=colors, opacity=0.8),
                    text=[f"t={e.get('frame_idx')}" for e in events if e.get("embedding_snippet")],
                    hovertemplate="%{text}<extra></extra>",
                ))
            except Exception:
                pass
    umap_fig.update_layout(title="Frame Embeddings (green=consensus ACT)", **dark)

    # ── Agent accuracy bars ─────────────────────────────────────────────────
    agent_names = [a.name for a in DEFAULT_AGENTS]
    correct_counts = {n: 0 for n in agent_names}
    wrong_counts = {n: 0 for n in agent_names}
    for e in events:
        for av in e.get("agent_verdicts", []):
            n = av.get("agent_name", "")
            if n in correct_counts:
                if av.get("decision") == "ACT":
                    if av.get("correct"):
                        correct_counts[n] += 1
                    else:
                        wrong_counts[n] += 1

    bar_fig = go.Figure(layout=go.Layout(**dark))
    bar_fig.add_trace(go.Bar(
        name="Correct", x=agent_names,
        y=[correct_counts[n] for n in agent_names],
        marker_color="#3fb950",
    ))
    bar_fig.add_trace(go.Bar(
        name="Wrong", x=agent_names,
        y=[wrong_counts[n] for n in agent_names],
        marker_color="#f85149",
    ))
    bar_fig.update_layout(barmode="group", title="Agent ACT Decisions", **dark)

    # ── Trajectory ACT/THINK line ────────────────────────────────────────────
    line_fig = go.Figure(layout=go.Layout(**dark))
    if events:
        frames = [e.get("frame_idx", 0) for e in events]
        act_counts = [e.get("act_count", 0) for e in events]
        line_fig.add_trace(go.Scatter(
            x=frames, y=act_counts, mode="lines+markers",
            name="ACT votes", line=dict(color="#58a6ff"),
        ))
        line_fig.add_hline(
            y=2, line_dash="dash", line_color="#f85149",
            annotation_text="Consensus threshold",
        )
    line_fig.update_layout(title="ACT Votes Per Frame", **dark)

    # ── Event log ────────────────────────────────────────────────────────────
    log_lines = []
    for e in events[-30:]:
        traj = e.get("trajectory_id", "?")
        frame = e.get("frame_idx", "?")
        consensus = "✓ RELEASE" if e.get("consensus") else "· hold"
        log_lines.append(f"[{traj} t={frame}] {consensus} | ACT={e.get('act_count',0)} THINK={e.get('think_count',0)}")
    log_text = "\n".join(reversed(log_lines)) if log_lines else "Waiting for data..."

    return umap_fig, bar_fig, line_fig, log_text


# ── Telemetry writer (called by orchestrator) ────────────────────────────────

def push_telemetry_event(event: dict):
    """Push a telemetry event to Redis for the dashboard to pick up."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        raw = r.get("clasp:telemetry")
        events = json.loads(raw) if raw else []
        events.append(event)
        # Keep last 500 events
        if len(events) > 500:
            events = events[-500:]
        r.set("clasp:telemetry", json.dumps(events))
    except Exception:
        pass


if __name__ == "__main__":
    app.run(host=DASH_HOST, port=DASH_PORT, debug=False)
