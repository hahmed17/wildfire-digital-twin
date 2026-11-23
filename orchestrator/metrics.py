"""
§10 metrics.py — how good was the analysis, and what did the agents do.

Two families, kept separate because they answer different questions:

  accuracy   how close the analysis perimeter is to the observed one
             (IoU, area error, centroid offset) — the dependent variable
  behaviour  what each agent decided, how often guardrails overrode it,
             how much latency the agents added

No FARSITE and no `src/` imports beyond geometry, so a finished run can be
scored on any machine from its saved outputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon

# An innovation above this counts as the forecast disagreeing with the observation.
# Matches the threshold agents/observation.compute_signals uses for its streak count.
DISAGREEMENT_THRESHOLD = 1.5


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def state_to_polygon(state_vec) -> Polygon | None:
    """(2*n_vertex,) or (2*n_vertex, 1) state vector -> the largest valid polygon."""
    v = np.asarray(state_vec).ravel()
    if len(v) < 6:
        return None
    geom = make_valid(Polygon(np.c_[v[0::2], v[1::2]]))
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    parts = [p for g in getattr(geom, "geoms", [])
             for p in (g.geoms if isinstance(g, MultiPolygon) else [g])
             if isinstance(p, Polygon)]
    return max(parts, key=lambda p: p.area) if parts else None


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def iou(analysis: Polygon | None, observed: Polygon | None) -> float | None:
    """Intersection over union. 1.0 = identical, 0.0 = disjoint.

    The headline accuracy number: unlike area error it penalises a perimeter that
    is the right size in the wrong place, which is exactly the failure a bad
    observation or a drifting forecast produces.
    """
    if analysis is None or observed is None:
        return None
    union = analysis.union(observed).area
    if union <= 0:
        return None
    return float(analysis.intersection(observed).area / union)


def area_error(analysis: Polygon | None, observed: Polygon | None) -> dict[str, float | None]:
    """Signed and relative area error in m^2 / fraction.

    Signed, because over- and under-prediction mean different things
    operationally: an over-predicted perimeter wastes resources, an
    under-predicted one leaves people inside the fire.
    """
    if analysis is None or observed is None or observed.area <= 0:
        return {"area_error_m2": None, "area_error_frac": None,
                "analysis_area_m2": None, "observed_area_m2": None}
    err = analysis.area - observed.area
    return {"area_error_m2": float(err),
            "area_error_frac": float(err / observed.area),
            "analysis_area_m2": float(analysis.area),
            "observed_area_m2": float(observed.area)}


def centroid_offset_m(analysis: Polygon | None, observed: Polygon | None) -> float | None:
    """Distance between centroids (m). Catches a displaced perimeter that happens
    to have about the right area."""
    if analysis is None or observed is None:
        return None
    a, o = analysis.centroid, observed.centroid
    return float(math.hypot(a.x - o.x, a.y - o.y))


def score_perimeter(analysis, observed) -> dict[str, Any]:
    """All accuracy measures for one cycle. Accepts polygons or state vectors."""
    pa = analysis if isinstance(analysis, Polygon) else state_to_polygon(analysis)
    po = observed if isinstance(observed, Polygon) else state_to_polygon(observed)
    out: dict[str, Any] = {"iou": iou(pa, po),
                           "centroid_offset_m": centroid_offset_m(pa, po)}
    out.update(area_error(pa, po))
    return out


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

def decision_rows(log: Iterable) -> list[dict[str, Any]]:
    """Flatten a run's DecisionRecord list into rows for a CSV.

    One row per decision, so a run's decisions can be counted, grouped by agent,
    and compared across arms without reopening the pickle.
    """
    rows = []
    for rec in log:
        d = rec.decision if isinstance(rec.decision, dict) else {}
        rows.append({
            "sim_time": rec.sim_time.isoformat() if hasattr(rec.sim_time, "isoformat")
                        else str(rec.sim_time),
            "node": rec.node,
            "policy": rec.policy,
            "action": d.get("action"),
            "reason": d.get("reason", ""),
            "overridden_by_guardrail": bool(rec.overridden_by_guardrail),
            "agent_latency_s": float(rec.agent_latency_s),
            "tokens": int(getattr(rec, "tokens", 0) or 0),
            # the fields a policy most often keys on, when present
            "vsize_m": d.get("vsize_m"),
            "wsize_m": d.get("wsize_m"),
            "inflation": d.get("inflation"),
        })
    return rows


def behaviour_summary(log: Iterable) -> dict[str, Any]:
    """Decision counts per agent, guardrail overrides, agent latency and tokens.

    Only nodes that actually choose something are counted. `forecast` and
    `synthesize` log records for provenance but take no decision, so they stay in
    decisions.csv and out of the counts — otherwise the table grows a meaningless
    "forecast: None" row.
    """
    rows = decision_rows(log)
    per_node: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["action"] is None:
            continue
        per_node.setdefault(r["node"], {})
        key = str(r["action"])
        per_node[r["node"]][key] = per_node[r["node"]].get(key, 0) + 1

    n = len(rows)
    overrides = sum(r["overridden_by_guardrail"] for r in rows)
    return {
        "decisions_total": n,
        "decisions_per_node": per_node,
        "guardrail_overrides": overrides,
        "guardrail_override_rate": (overrides / n) if n else None,
        "agent_latency_s_total": sum(r["agent_latency_s"] for r in rows),
        "agent_latency_s_max": max((r["agent_latency_s"] for r in rows), default=0.0),
        "tokens_total": sum(r["tokens"] for r in rows),
        "policies": sorted({r["policy"] for r in rows}),
    }


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------

def score_run(final_state: dict, observed_states: dict[int, Any] | None = None,
              analysis_states: dict[int, Any] | None = None) -> dict[str, Any]:
    """Score a completed replay.

    Accuracy needs the perimeters, which TwinState does not carry (it holds paths).
    `replay.py` collects them per cycle and passes them in; without them the result
    still carries the behaviour summary and whatever areas `update` recorded.

    Args:
        final_state: the TwinState returned by run_pipeline
        observed_states: {step: observed state vector or polygon}
        analysis_states: {step: analysis state vector or polygon}

    Returns:
        {"cycles": [...], "accuracy": {...}, "behaviour": {...}, "drift_history": [...]}
    """
    cycles = []
    for m in final_state.get("metrics", []):
        row = dict(m)
        step = m.get("step")
        if observed_states and analysis_states and step in observed_states \
                and step in analysis_states:
            row.update(score_perimeter(analysis_states[step], observed_states[step]))
        cycles.append(row)

    ious = [c["iou"] for c in cycles if c.get("iou") is not None]
    aerr = [abs(c["area_error_frac"]) for c in cycles
            if c.get("area_error_frac") is not None]
    offs = [c["centroid_offset_m"] for c in cycles
            if c.get("centroid_offset_m") is not None]

    accuracy = {
        "n_scored_cycles": len(ious),
        "iou_mean": float(np.mean(ious)) if ious else None,
        "iou_min": float(np.min(ious)) if ious else None,
        "iou_final": ious[-1] if ious else None,
        "abs_area_error_frac_mean": float(np.mean(aerr)) if aerr else None,
        "centroid_offset_m_mean": float(np.mean(offs)) if offs else None,
    }

    drift = list(final_state.get("drift_history") or [])
    return {
        "cycles": cycles,
        "accuracy": accuracy,
        "behaviour": behaviour_summary(final_state.get("log", [])),
        "drift_history": drift,
        "disagreements": sum(1 for v in drift if v > DISAGREEMENT_THRESHOLD),
    }


def write_run_metrics(out_dir: str | Path, scored: dict[str, Any],
                      final_state: dict | None = None) -> dict[str, Path]:
    """Write metrics.json and decisions.csv into `out_dir`. Returns the paths."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    mpath = out_dir / "metrics.json"
    mpath.write_text(json.dumps(scored, indent=2, default=str))
    paths["metrics"] = mpath

    if final_state is not None:
        rows = decision_rows(final_state.get("log", []))
        if rows:
            dpath = out_dir / "decisions.csv"
            with open(dpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            paths["decisions"] = dpath

    return paths


# ---------------------------------------------------------------------------
# Comparing arms
# ---------------------------------------------------------------------------

def compare_runs(runs: dict[str, dict[str, Any]]):
    """One row per arm, for the paper's results table.

    Args:
        runs: {arm_label: scored dict from score_run or a loaded metrics.json}

    Returns:
        pandas.DataFrame indexed by arm, or a list of dicts if pandas is absent.
    """
    rows = []
    for label, scored in runs.items():
        acc = scored.get("accuracy", {})
        beh = scored.get("behaviour", {})
        actions: dict[str, int] = {}
        for node, counts in (beh.get("decisions_per_node") or {}).items():
            for action, k in counts.items():
                actions[f"{node}:{action}"] = k
        rows.append({
            "arm": label,
            "cycles": len(scored.get("cycles", [])),
            "iou_mean": acc.get("iou_mean"),
            "iou_final": acc.get("iou_final"),
            "abs_area_err": acc.get("abs_area_error_frac_mean"),
            "centroid_m": acc.get("centroid_offset_m_mean"),
            "guardrails": beh.get("guardrail_overrides"),
            "agent_s": round(beh.get("agent_latency_s_total") or 0.0, 3),
            "tokens": beh.get("tokens_total"),
            **actions,
        })
    try:
        import pandas as pd
        return pd.DataFrame(rows).set_index("arm")
    except ImportError:
        return rows


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load a metrics.json written by write_run_metrics."""
    return json.loads(Path(path).read_text())
