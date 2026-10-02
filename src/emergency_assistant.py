"""
Emergency assistant: turns a simulation result into an operator briefing.

Design note (honest about what this is):
  * By default this is a DETERMINISTIC REASONING ENGINE, not a language model.
    It reads the actual simulation numbers (destroyed junctions, surviving
    connectivity, isolated facilities, post-event critical nodes) and composes a
    structured situation report with concrete priorities.  It works offline,
    costs nothing, and always cites real numbers from the run.
  * If an LLM API key is available (OPENAI_API_KEY etc.) the same briefing facts
    can be sent to that model to phrase the report naturally.  Set EMERGENCY_LLM=1
    to enable.  Without a key the local engine is used and the app is fully
    functional.

The engine deliberately separates SITUATION / IMPACT / PRIORITY.  Facts are
always computed locally - a language model only ever writes the prose, so it
cannot invent numbers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEVERITY_BANDS = [
    (90.0, "MINOR", "green",
     "The network absorbs this event. Normal response posture."),
    (60.0, "MODERATE", "amber",
     "Degradation is contained but response times will increase in the "
     "affected corridor."),
    (25.0, "SEVERE", "orange",
     "Large parts of the network are cut off. Emergency access is the "
     "priority."),
    (0.0, "CRITICAL", "red",
     "The serviceable network has collapsed. Only a fraction of junctions "
     "remain mutually reachable."),
]


def classify_severity(connectivity_pct: float, destroyed_facilities: int = 0) -> tuple:
    """Severity from surviving connectivity, escalated if facilities were hit.

    A network can keep 100% of its connectivity while a hospital sits inside
    the blast radius, so destroyed facilities escalate the level rather than
    being reported as a footnote.
    """
    label, colour, note = ("CRITICAL", "red", SEVERITY_BANDS[-1][3])
    for threshold, lbl, col, nt in SEVERITY_BANDS:
        if connectivity_pct >= threshold:
            label, colour, note = lbl, col, nt
            break
    if destroyed_facilities and label == "MINOR":
        label, colour = "SEVERE", "orange"
        note = ("The network survives, but a response facility was destroyed - "
                "local access, not network connectivity, is the problem.")
    return label, colour, note


def build_briefing(scenario: dict, damage: dict, graph_stats: dict) -> dict:
    """Compose the situation report from the real simulation numbers."""
    conn = damage["connectivity_pct"]
    destroyed_fac = [f["name"] for f in damage["facilities"] if f["destroyed"]]
    severity, colour, sev_note = classify_severity(conn, len(destroyed_fac))
    lost_nodes = damage["nodes_before"] - damage["nodes_after"]
    newly = damage.get("newly_isolated_facilities", [])
    survivors = [f["name"] for f in damage["facilities"]
                 if f["reachable"] and not f["destroyed"]]
    top = damage.get("post_centrality_top", [])
    top_node = top[0]["node"] if top else None
    top_delta = top[0]["delta"] if top else 0.0

    situation = [
        f"{scenario['name']} simulated on the {graph_stats.get('nodes', 0)}-junction "
        f"extracted road network.",
        f"{lost_nodes} junctions destroyed and removed from the routable graph.",
        f"Connected components {damage['components_before']} -> "
        f"{damage['components_after']}; the largest mutually-reachable network "
        f"holds {damage['largest_after']} of {damage['largest_before']} junctions "
        f"({conn}% of pre-event connectivity).",
        f"{severity} - {sev_note}",
    ]

    impact = []
    if newly:
        impact.append(f"NEWLY isolated facilities: {', '.join(newly)}.")
    else:
        impact.append("No facility that was connected before is newly cut off.")
    if destroyed_fac:
        impact.append(f"Facilities destroyed outright: {', '.join(destroyed_fac)}.")
    if survivors:
        impact.append("Still reachable from the surviving network: "
                      + ", ".join(survivors) + ".")
    impact.append(f"Serviceable network retains {damage['serviceable_nodes']} of "
                  f"{damage['nodes_before']} junctions.")

    priority = []
    step = 1
    if newly:
        priority.append(
            f"{step}. Establish access to {newly[0]} - it was connected before "
            f"this event and is now cut off.")
        step += 1
    if top_node is not None and top_delta > 0:
        priority.append(
            f"{step}. Junction {top_node} is now the most load-bearing point in "
            f"the surviving network (centrality rose by {top_delta:.3f}); protect "
            f"it before any other single junction.")
        step += 1
    if destroyed_fac:
        priority.append(
            f"{step}. {destroyed_fac[0]} is physically destroyed - triage there "
            f"cannot be routed until a route is restored.")
        step += 1
    priority.append(
        f"{step}. Re-run with a smaller hazard radius to find the minimum damage "
        f"that still isolates a facility (the response-planning threshold).")
    if conn >= 90:
        priority.append(
            f"{step + 1}. Maintain normal patrols: the network has enough "
            f"redundancy to absorb single-junction failures.")

    return {
        "severity": severity,
        "colour": colour,
        "connectivity_pct": conn,
        "situation": situation,
        "impact": impact,
        "priority": priority,
        "headline": (f"{severity}: {lost_nodes} junctions lost, {conn}% of the "
                     f"network still connected"),
    }


# ---------------------------------------------------------------------------
# Optional LLM phrasing (never required)
# ---------------------------------------------------------------------------

def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or
                os.environ.get("GEMINI_API_KEY") or
                os.environ.get("ANTHROPIC_API_KEY"))


def llm_phrase(briefing: dict):
    """Optionally rephrase the briefing with a hosted LLM, if a key is set.

    Returns None when disabled / unavailable, in which case the deterministic
    briefing is used unchanged.  The JSON facts are computed locally, so a model
    can only change the wording, never the numbers.
    """
    if os.environ.get("EMERGENCY_LLM", "0") not in ("1", "true", "True"):
        return None
    if not llm_available():
        return None
    facts = json.dumps({k: briefing[k] for k in
                        ("headline", "situation", "impact", "priority")}, indent=2)
    prompt = ("You are an emergency operations assistant. Using ONLY the JSON "
              "facts below (never invent numbers), write a concise radio-style "
              "briefing with three short sections: SITUATION, IMPACT, PRIORITY.\n\n"
              + facts)
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read().decode())
        return out["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        return f"(LLM unavailable: {exc})"
