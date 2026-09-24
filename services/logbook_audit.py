"""Deterministic logbook audit — scans every entry for errors, inconsistencies,
and duplicates and returns structured findings. Cheap, reliable, runs over the
whole logbook; the AI assistant layers judgment on top and applies fixes."""
import datetime
from collections import defaultdict


def _f(v):
    try:
        if v in (None, ""):
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _slim(e):
    return {
        "id": e.get("id"),
        "date": e.get("date", ""),
        "tail": e.get("aircraft_ident", ""),
        "model": e.get("aircraft_model", ""),
        "route": f"{e.get('route_from', '')}-{e.get('route_to', '')}",
        "total": _f(e.get("total_duration")),
        "day": _f(e.get("day")),
        "night": _f(e.get("night")),
        "source": e.get("source", ""),
        "reviewed": bool(e.get("reviewed")),
        "locked": bool(e.get("locked")),
        "remarks": (e.get("remarks", "") or "")[:80],
    }


def audit_entries(entries):
    findings = []
    counter = {"n": 0}

    def nf():
        counter["n"] += 1
        return f"F{counter['n']}"

    # ---- duplicates: same date + from + to ----
    groups = defaultdict(list)
    for e in entries:
        d = e.get("date", "")
        rf = (e.get("route_from", "") or "").upper()
        rt = (e.get("route_to", "") or "").upper()
        if d and (rf or rt):
            groups[(d, rf, rt)].append(e)
    for (d, rf, rt), grp in groups.items():
        if len(grp) < 2:
            continue

        def completeness(e):
            return sum(1 for v in e.values() if v not in (None, "", 0, 0.0, False))

        keep = sorted(grp, key=lambda e: (bool(e.get("locked")), bool(e.get("reviewed")), completeness(e)), reverse=True)[0]
        dupes = [e for e in grp if e.get("id") != keep.get("id")]
        srcs = sorted({(e.get("source") or "?") for e in grp})
        findings.append({
            "id": nf(), "type": "duplicate", "severity": "high",
            "entries": [_slim(e) for e in grp],
            "explanation": f"{len(grp)} entries for {d} {rf}→{rt}" + (f" (sources: {', '.join(srcs)})" if len(srcs) > 1 else ""),
            "suggested_fix": {"action": "merge", "keep_id": keep.get("id"), "delete_ids": [e.get("id") for e in dupes]},
        })

    # ---- per-entry checks ----
    today = datetime.date.today()
    for e in entries:
        total = _f(e.get("total_duration"))
        day, night = _f(e.get("day")), _f(e.get("night"))
        inst = _f(e.get("actual_inst")) + _f(e.get("simulated_inst"))
        xc = _f(e.get("cross_country"))
        rf = (e.get("route_from", "") or "").upper()
        rt = (e.get("route_to", "") or "").upper()

        if total > 0 and (day > 0 or night > 0) and abs((day + night) - total) > 0.2:
            findings.append({"id": nf(), "type": "time_mismatch", "severity": "medium", "entries": [_slim(e)],
                             "explanation": f"day {day} + night {night} = {round(day + night, 1)}, but total is {total}",
                             "suggested_fix": {"action": "review", "id": e.get("id"), "note": "day + night should equal total time"}})
        for label, val in (("night", night), ("instrument", inst), ("cross-country", xc)):
            if total > 0 and val > total + 0.05:
                findings.append({"id": nf(), "type": "impossible_value", "severity": "high", "entries": [_slim(e)],
                                 "explanation": f"{label} time {round(val, 1)} exceeds total time {total}",
                                 "suggested_fix": {"action": "review", "id": e.get("id")}})
        if total < 0 or day < 0 or night < 0:
            findings.append({"id": nf(), "type": "negative_value", "severity": "high", "entries": [_slim(e)],
                             "explanation": "negative time value", "suggested_fix": {"action": "review", "id": e.get("id")}})

        d = _parse_date(e.get("date", ""))
        if d:
            if d > today:
                findings.append({"id": nf(), "type": "future_date", "severity": "high", "entries": [_slim(e)],
                                 "explanation": f"date {e.get('date')} is in the future",
                                 "suggested_fix": {"action": "review", "id": e.get("id")}})
            elif d.year < 1930:
                findings.append({"id": nf(), "type": "bad_date", "severity": "medium", "entries": [_slim(e)],
                                 "explanation": f"date {e.get('date')} looks wrong (year {d.year})",
                                 "suggested_fix": {"action": "review", "id": e.get("id")}})

        if rf and rf == rt and xc > 0:
            findings.append({"id": nf(), "type": "xc_same_airport", "severity": "low", "entries": [_slim(e)],
                             "explanation": f"cross-country {xc} logged, but from and to are both {rf}",
                             "suggested_fix": {"action": "review", "id": e.get("id")}})

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda x: order.get(x["severity"], 3))
    return findings
