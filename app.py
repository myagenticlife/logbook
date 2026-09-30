"""
Pilot Logbook - Web Application

A web-based pilot logbook using ASA Standard format with SQLite storage.
"""

import os
import time
from datetime import datetime

from authlib.integrations.flask_client import OAuth
from config import Config
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from flask_login import LoginManager, current_user, login_required, login_user
import jwt as _jwt
from models.logbook_entry import Logbook, LogbookEntry, make_entry_key, normalize_airport
from models.user import User as _SuiteUser, generate_access_token as _gen_token
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config.from_object(Config)
app.config["PREFERRED_URL_SCHEME"] = "https"
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Pre-load airport/runway CSV caches at startup so first scan isn't slow
try:
    from services.airport_lookup import _load_icao_map
    _load_icao_map()
    from services.airport_approaches import _get_runways
    _get_runways("")  # triggers _load_runway_data()
except Exception as e:
    print(f"Warning: failed to pre-load airport data: {e}")

# Use PostgreSQL if DATABASE_URL is set, otherwise fall back to SQLite
if Config.DATABASE_URL:
    from services.postgres_storage import PostgresStorage
    storage = PostgresStorage(database_url=Config.DATABASE_URL)
    print(f"Using PostgreSQL database")
else:
    from services.sqlite_storage import SQLiteStorage
    storage = SQLiteStorage(db_path=Config.SQLITE_DB_PATH)
    print(f"Using SQLite database: {Config.SQLITE_DB_PATH}")

# ── Auth setup ─────────────────────────────────────────────────
from datetime import timedelta

app.permanent_session_lifetime = timedelta(days=90)
app.config["REMEMBER_COOKIE_DURATION"] = timedelta(days=90)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.access"


@login_manager.user_loader
def load_user(user_id):
    return storage.get_user(user_id)


@app.before_request
def _suite_sso():
    """Single sign-on with the MyAgenticLife suite: if the visitor carries a
    valid suite session cookie (mal_session JWT), sign them into the logbook
    automatically so they never hit the local token wall. Inert (no-op) unless
    SESSION_SECRET is configured to match the accounts service."""
    if current_user.is_authenticated:
        return
    token = request.cookies.get("mal_session")
    secret = os.environ.get("SESSION_SECRET")
    if not token or not secret:
        return
    try:
        payload = _jwt.decode(token, secret, algorithms=["HS256"])
    except Exception:
        return
    email = (payload.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return  # e.g. the admin-code session (email 'admin') — ignore
    user = storage.get_user_by_email(email)
    if user is None:
        user = _SuiteUser(email=email, name=(payload.get("name") or email.split("@")[0]), access_token=_gen_token())
        storage.create_user(user)
    login_user(user, remember=True)


# Google OAuth client kept registered for future use; UI is hidden.
oauth = OAuth(app)
if Config.GOOGLE_OAUTH_CLIENT_ID:
    oauth.register(
        name="google",
        client_id=Config.GOOGLE_OAUTH_CLIENT_ID,
        client_secret=Config.GOOGLE_OAUTH_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={
            "scope": "openid email profile https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/drive.file",
        },
        authorize_params={
            "access_type": "offline",
            "prompt": "consent",
        },
    )

# Register auth blueprint
from auth import auth_bp, ensure_admin_bootstrap, is_admin  # noqa: E402

app.register_blueprint(auth_bp)
app.jinja_env.globals["is_admin"] = is_admin

# Ensure admin user exists with a valid token; print token to logs if generated.
# Run OFF the import path in a background thread: a slow/blocked first DB
# connection must never stop gunicorn from binding the port (Render kills the
# deploy with "no open ports" if import blocks). The port binds immediately;
# this finishes in the background.
import threading as _threading

_threading.Thread(target=ensure_admin_bootstrap, args=(storage,), daemon=True).start()


def parse_float(value, default=0.0):
    """Safely parse a float value."""
    try:
        return float(value) if value else default
    except (ValueError, TypeError):
        return default


def parse_int(value, default=0):
    """Safely parse an int value."""
    try:
        return int(value) if value else default
    except (ValueError, TypeError):
        return default


def normalize_route_via(route_via: str) -> str:
    """Normalize a multi-leg route string: SQL-PAE-BFI → KSQL-KPAE-KBFI."""
    if not route_via:
        return ""
    legs = [normalize_airport(leg.strip()) for leg in route_via.split("-") if leg.strip()]
    return "-".join(legs)


@app.route("/")
def index():
    """The suite front door now lives at myagenticlife.com — send people there
    instead of the retired local 'Tools for the modern pilot' landing."""
    return redirect("https://myagenticlife.com")


@app.route("/logbook")
@login_required
def logbook_view():
    """Main logbook view - summary list."""
    uid = current_user.id
    entries = storage.get_all_entries(user_id=uid, sort_by_date=True)
    totals = storage.get_totals(user_id=uid)
    backup_sheet_url = ""
    if current_user.backup_sheet_id:
        backup_sheet_url = f"https://docs.google.com/spreadsheets/d/{current_user.backup_sheet_id}"
    return render_template(
        "logbook.html",
        entries=entries,
        totals=totals,
        backup_sheet_url=backup_sheet_url,
    )


@app.route("/entry/new")
@login_required
def new_entry():
    """New flight entry page with defaults."""
    defaults = {
        "id": "",
        "date": datetime.now().strftime("%m/%d/%Y"),
        "aircraft_model": current_user.default_aircraft_type or Config.DEFAULT_AIRCRAFT_TYPE,
        "aircraft_ident": current_user.default_tail_number or Config.DEFAULT_TAIL_NUMBER,
        "route_from": current_user.default_departure or Config.DEFAULT_DEPARTURE,
        "route_to": "",
        "sel": "",
        "mel": "",
        "day": "",
        "night": "",
        "cross_country": "",
        "actual_inst": "",
        "simulated_inst": "",
        "num_inst_app": "",
        "landings_day": "",
        "landings_night": "",
        "pic": "",
        "sic": "",
        "dual_recd": "",
        "dual_given": "",
        "solo": "",
        "sim": "",
        "total_duration": "",
        "remarks": "",
    }
    return render_template("entry.html", entry=defaults, is_new=True)


@app.route("/entry/<entry_id>")
@login_required
def view_entry(entry_id):
    """View/edit a specific flight entry."""
    entry = storage.get_entry(entry_id, user_id=current_user.id)
    if not entry:
        return redirect(url_for("logbook_view"))
    return render_template("entry.html", entry=entry.to_dict(), is_new=False)


@app.route("/api/entries", methods=["GET"])
@login_required
def get_entries():
    """Get all logbook entries."""
    entries = storage.get_all_entries(user_id=current_user.id, sort_by_date=True)
    return jsonify([e.to_dict() for e in entries])


@app.route("/api/entries/<entry_id>", methods=["GET"])
@login_required
def get_entry(entry_id):
    """Get a specific logbook entry."""
    entry = storage.get_entry(entry_id, user_id=current_user.id)
    if entry:
        return jsonify(entry.to_dict())
    return jsonify({"error": "Entry not found"}), 404


@app.route("/api/entries", methods=["POST"])
@login_required
def create_entry():
    """Create a new logbook entry."""
    data = request.json
    entry = LogbookEntry(
        date=data.get("date", ""),
        aircraft_model=data.get("aircraft_model", ""),
        aircraft_ident=data.get("aircraft_ident", "").upper(),
        route_from=normalize_airport(data.get("route_from", "")),
        route_to=normalize_airport(data.get("route_to", "")),
        route_via=normalize_route_via(data.get("route_via", "")),
        sel=parse_float(data.get("sel")),
        mel=parse_float(data.get("mel")),
        day=parse_float(data.get("day")),
        night=parse_float(data.get("night")),
        cross_country=parse_float(data.get("cross_country")),
        actual_inst=parse_float(data.get("actual_inst")),
        simulated_inst=parse_float(data.get("simulated_inst")),
        num_inst_app=parse_int(data.get("num_inst_app")),
        landings_day=parse_int(data.get("landings_day")),
        landings_night=parse_int(data.get("landings_night")),
        takeoffs_day=parse_int(data.get("takeoffs_day")),
        takeoffs_night=parse_int(data.get("takeoffs_night")),
        pic=parse_float(data.get("pic")),
        sic=parse_float(data.get("sic")),
        dual_recd=parse_float(data.get("dual_recd")),
        dual_given=parse_float(data.get("dual_given")),
        solo=parse_float(data.get("solo")),
        sim=parse_float(data.get("sim")),
        total_duration=parse_float(data.get("total_duration")),
        remarks=data.get("remarks", ""),
        source=data.get("source", "manual"),
    )

    storage.add_entry(entry, user_id=current_user.id)
    return jsonify({"id": entry.id, "message": "Entry created"}), 201


@app.route("/api/entries/<entry_id>", methods=["PUT"])
@login_required
def update_entry(entry_id):
    """Update an existing logbook entry."""
    uid = current_user.id
    entry = storage.get_entry(entry_id, user_id=uid)
    if not entry:
        return jsonify({"error": "Entry not found"}), 404

    data = request.json

    entry.date = data.get("date", entry.date)
    entry.aircraft_model = data.get("aircraft_model", entry.aircraft_model)
    entry.aircraft_ident = data.get("aircraft_ident", entry.aircraft_ident).upper()
    entry.route_from = normalize_airport(data.get("route_from", entry.route_from))
    entry.route_to = normalize_airport(data.get("route_to", entry.route_to))
    entry.route_via = normalize_route_via(data.get("route_via", entry.route_via))
    entry.sel = parse_float(data.get("sel"), entry.sel)
    entry.mel = parse_float(data.get("mel"), entry.mel)
    entry.day = parse_float(data.get("day"), entry.day)
    entry.night = parse_float(data.get("night"), entry.night)
    entry.cross_country = parse_float(data.get("cross_country"), entry.cross_country)
    entry.actual_inst = parse_float(data.get("actual_inst"), entry.actual_inst)
    entry.simulated_inst = parse_float(data.get("simulated_inst"), entry.simulated_inst)
    entry.num_inst_app = parse_int(data.get("num_inst_app"), entry.num_inst_app)
    entry.landings_day = parse_int(data.get("landings_day"), entry.landings_day)
    entry.landings_night = parse_int(data.get("landings_night"), entry.landings_night)
    entry.takeoffs_day = parse_int(data.get("takeoffs_day"), entry.takeoffs_day)
    entry.takeoffs_night = parse_int(data.get("takeoffs_night"), entry.takeoffs_night)
    entry.pic = parse_float(data.get("pic"), entry.pic)
    entry.sic = parse_float(data.get("sic"), entry.sic)
    entry.dual_recd = parse_float(data.get("dual_recd"), entry.dual_recd)
    entry.dual_given = parse_float(data.get("dual_given"), entry.dual_given)
    entry.solo = parse_float(data.get("solo"), entry.solo)
    entry.sim = parse_float(data.get("sim"), entry.sim)
    entry.total_duration = parse_float(data.get("total_duration"), entry.total_duration)
    entry.remarks = data.get("remarks", entry.remarks)

    storage.update_entry(entry, user_id=uid)
    return jsonify({"message": "Entry updated"})


@app.route("/api/entries/<entry_id>", methods=["DELETE"])
@login_required
def delete_entry(entry_id):
    """Delete a logbook entry. Refuses to delete locked entries."""
    uid = current_user.id
    entry = storage.get_entry(entry_id, user_id=uid)
    if not entry:
        return jsonify({"error": "Entry not found"}), 404
    if entry.locked:
        return jsonify({"error": "Entry is locked and cannot be deleted"}), 403
    if storage.delete_entry(entry_id, user_id=uid):
        return jsonify({"message": "Entry deleted"})
    return jsonify({"error": "Delete failed"}), 500


@app.route("/api/audit", methods=["GET"])
@login_required
def audit_logbook_api():
    """Deterministic audit of the signed-in user's logbook — returns structured
    findings (duplicates, time mismatches, impossible values, bad dates)."""
    from services.logbook_audit import audit_entries
    entries = storage.get_all_entries(user_id=current_user.id, sort_by_date=True)
    dicts = [e.to_dict() for e in entries]
    findings = audit_entries(dicts)
    return jsonify({"total_entries": len(dicts), "finding_count": len(findings), "findings": findings})


@app.route("/api/entries/batch", methods=["DELETE"])
@login_required
def batch_delete_entries():
    """Delete multiple logbook entries, skipping locked ones."""
    data = request.json
    entry_ids = data.get("entry_ids", [])

    if not entry_ids:
        return jsonify({"error": "No entry IDs provided"}), 400

    result = storage.delete_entries(entry_ids, user_id=current_user.id)
    msg = f"Deleted {result['deleted']} entries"
    if result["skipped_locked"]:
        msg += f" ({result['skipped_locked']} locked entries skipped)"

    return jsonify({
        "success": True,
        "deleted": result["deleted"],
        "skipped_locked": result["skipped_locked"],
        "message": msg,
    })


@app.route("/api/entries/<entry_id>/lock", methods=["POST"])
@login_required
def toggle_lock(entry_id):
    """Toggle the locked state of an entry."""
    data = request.json or {}
    locked = data.get("locked", True)
    if storage.toggle_entry_field(entry_id, "locked", locked, user_id=current_user.id):
        return jsonify({"success": True, "locked": locked})
    return jsonify({"error": "Entry not found"}), 404


@app.route("/api/entries/<entry_id>/review", methods=["POST"])
@login_required
def toggle_review(entry_id):
    """Toggle the reviewed state of an entry."""
    data = request.json or {}
    reviewed = data.get("reviewed", True)
    if storage.toggle_entry_field(entry_id, "reviewed", reviewed, user_id=current_user.id):
        return jsonify({"success": True, "reviewed": reviewed})
    return jsonify({"error": "Entry not found"}), 404


@app.route("/api/totals", methods=["GET"])
@login_required
def get_totals():
    """Get logbook totals."""
    return jsonify(storage.get_totals(user_id=current_user.id))


# ---- Agent query layer: safe, parameterized filtering over the user's own entries ----
_QUERY_FIELDS = {
    "date", "aircraft_model", "aircraft_ident", "route_from", "route_to", "route_via",
    "sel", "mel", "day", "night", "cross_country", "actual_inst", "simulated_inst",
    "num_inst_app", "landings_day", "landings_night", "takeoffs_day", "takeoffs_night",
    "pic", "sic", "dual_recd", "dual_given", "solo", "sim", "total_duration",
    "remarks", "source", "locked", "reviewed", "id",
}
_NUM_FIELDS = {
    "sel", "mel", "day", "night", "cross_country", "actual_inst", "simulated_inst",
    "num_inst_app", "landings_day", "landings_night", "takeoffs_day", "takeoffs_night",
    "pic", "sic", "dual_recd", "dual_given", "solo", "sim", "total_duration",
}


def _q_date(x):
    import datetime as _dt
    for f in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return _dt.datetime.strptime(str(x).strip(), f).date()
        except Exception:
            pass
    return None


def _q_num(x):
    try:
        return float(x or 0)
    except Exception:
        return 0.0


def _cond_matches(entry, cond):
    """Evaluate one {field, op, value} condition against an entry dict. Safe: field must
    be whitelisted; op is enumerated; comparison is type-aware (no eval, no SQL)."""
    field = cond.get("field")
    op = (cond.get("op") or "eq").lower()
    val = cond.get("value")
    if field not in _QUERY_FIELDS:
        raise ValueError(f"unknown field '{field}'")
    raw = entry.get(field)
    # date-specific operators
    if op in ("year", "month", "between", "before", "after"):
        d = _q_date(raw)
        if d is None:
            return False
        if op == "year":
            return d.year == int(val)
        if op == "month":
            return d.month == int(val)
        if op == "before":
            return d < _q_date(val)
        if op == "after":
            return d > _q_date(val)
        if op == "between":
            lo, hi = _q_date(val[0]), _q_date(val[1])
            return (lo is None or d >= lo) and (hi is None or d <= hi)
    # emptiness / zero
    if op == "is_empty":
        return raw in (None, "", 0, 0.0, "0")
    if op == "is_zero":
        return _q_num(raw) == 0.0
    if op == "not_empty":
        return raw not in (None, "", 0, 0.0, "0")
    if op == "contains":
        return str(val).lower() in str(raw or "").lower()
    if op == "in":
        vals = [str(v).lower() for v in (val or [])]
        return str(raw or "").lower() in vals
    # numeric vs string comparisons
    if field in _NUM_FIELDS:
        a, b = _q_num(raw), _q_num(val)
    else:
        a, b = str(raw or "").lower(), str(val or "").lower()
    return {
        "eq": a == b, "ne": a != b, "lt": a < b, "lte": a <= b,
        "gt": a > b, "gte": a >= b,
    }.get(op, False)


def _entry_matches(entry, where, logic):
    if not where:
        return True
    results = [_cond_matches(entry, c) for c in where]
    return all(results) if (logic or "and").lower() == "and" else any(results)


@app.route("/api/query", methods=["POST"])
@login_required
def query_logbook():
    """Filter the signed-in user's entries with a safe, parameterized grammar.
    Body: {where:[{field,op,value}...], logic:'and'|'or', fields:[...], sort, desc,
    limit, aggregate}. aggregate=true returns summed hours + count instead of rows.
    Reads only; scoped to current_user; no raw SQL."""
    data = request.json or {}
    where = data.get("where", [])
    logic = data.get("logic", "and")
    entries = [e.to_dict() for e in storage.get_all_entries(user_id=current_user.id)]
    try:
        matched = [e for e in entries if _entry_matches(e, where, logic)]
    except ValueError as ex:
        return jsonify({"error": str(ex), "allowed_fields": sorted(_QUERY_FIELDS)}), 400

    if data.get("aggregate"):
        agg = {f: 0.0 for f in _NUM_FIELDS}
        for e in matched:
            for f in _NUM_FIELDS:
                agg[f] += _q_num(e.get(f))
        return jsonify({"count": len(matched),
                        "totals": {f: round(v, 1) for f, v in agg.items()}})

    # projection + sort + limit
    default_fields = ["id", "date", "aircraft_ident", "aircraft_model",
                      "route_from", "route_to", "route_via", "total_duration",
                      "num_inst_app", "source"]
    fields = data.get("fields") or default_fields
    fields = [f for f in fields if f in _QUERY_FIELDS]
    sort = data.get("sort")
    if sort in _QUERY_FIELDS:
        keyfn = (_q_date if sort == "date" else (_q_num if sort in _NUM_FIELDS else (lambda e: str(e.get(sort) or "").lower())))
        matched.sort(key=lambda e: (keyfn(e.get("date")) or __import__("datetime").date(1900, 1, 1)) if sort == "date" else keyfn(e), reverse=bool(data.get("desc")))
    total = len(matched)
    limit = int(data.get("limit") or 200)
    rows = [{f: e.get(f) for f in fields} for e in matched[:limit]]
    return jsonify({"count": total, "returned": len(rows),
                    "truncated": total > len(rows), "entries": rows})


@app.route("/api/entries/batch", methods=["PUT"])
@login_required
def batch_update_entries():
    """Guarded mass-update. Body: {entry_ids:[...] OR where:[...]+logic, fields:{...},
    dry_run}. dry_run=true returns the affected count + sample without changing anything.
    Skips locked entries. Scoped to current_user."""
    data = request.json or {}
    fields = data.get("fields", {})
    uid = current_user.id
    # resolve target set
    if data.get("entry_ids"):
        targets = [storage.get_entry(i, user_id=uid) for i in data["entry_ids"]]
        targets = [t for t in targets if t]
    else:
        where = data.get("where", [])
        logic = data.get("logic", "and")
        all_e = storage.get_all_entries(user_id=uid)
        try:
            targets = [e for e in all_e if _entry_matches(e.to_dict(), where, logic)]
        except ValueError as ex:
            return jsonify({"error": str(ex)}), 400
    locked = [t for t in targets if getattr(t, "locked", False)]
    editable = [t for t in targets if not getattr(t, "locked", False)]

    if data.get("dry_run") or not fields:
        sample = [{"id": t.id, "date": t.date, "aircraft_ident": t.aircraft_ident,
                   "route_from": t.route_from, "route_to": t.route_to} for t in editable[:15]]
        return jsonify({"dry_run": True, "would_update": len(editable),
                        "skipped_locked": len(locked), "sample": sample,
                        "fields_to_set": fields})
    updated = 0
    allowed = _QUERY_FIELDS - {"id", "locked", "source"}
    for t in editable:
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k in _NUM_FIELDS:
                setattr(t, k, parse_float(v) if k not in ("num_inst_app", "landings_day", "landings_night", "takeoffs_day", "takeoffs_night") else parse_int(v))
            elif k in ("route_from", "route_to"):
                setattr(t, k, normalize_airport(v))
            elif k == "route_via":
                setattr(t, k, normalize_route_via(v))
            else:
                setattr(t, k, v)
        storage.update_entry(t, user_id=uid)
        updated += 1
    return jsonify({"success": True, "updated": updated, "skipped_locked": len(locked)})


@app.route("/api/summary", methods=["GET"])
@login_required
def get_summary():
    """Flexible, filtered totals so an agent can sum & checksum on demand.
    Query params (all optional): through=MM/DD/YYYY (on/before), from_date=MM/DD/YYYY (on/after),
    aircraft=IDENT, source=scan-book1|flightaware|... . Returns count + summed hours by category."""
    import datetime as _dt

    def _pdate(x):
        for f in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
            try:
                return _dt.datetime.strptime(str(x).strip(), f).date()
            except Exception:
                pass
        return None

    through = _pdate(request.args.get("through", "")) if request.args.get("through") else None
    from_date = _pdate(request.args.get("from_date", "")) if request.args.get("from_date") else None
    aircraft = (request.args.get("aircraft") or "").upper().strip()
    source = (request.args.get("source") or "").strip()

    entries = [e.to_dict() for e in storage.get_all_entries(user_id=current_user.id)]
    fields = ["total_duration", "sel", "mel", "day", "night", "cross_country",
              "actual_inst", "simulated_inst", "pic", "sic", "dual_recd", "dual_given", "solo", "sim"]
    sums = {f: 0.0 for f in fields}
    count = 0
    landings = 0
    matched_dates = []
    for e in entries:
        d = _pdate(e.get("date"))
        if through and (d is None or d > through):
            continue
        if from_date and (d is None or d < from_date):
            continue
        if aircraft and (e.get("aircraft_ident") or "").upper() != aircraft:
            continue
        if source and (e.get("source") or "") != source:
            continue
        count += 1
        if d:
            matched_dates.append(d)
        for f in fields:
            try:
                sums[f] += float(e.get(f) or 0)
            except Exception:
                pass
        landings += int(e.get("landings_day") or 0) + int(e.get("landings_night") or 0)
    return jsonify({
        "count": count,
        "total_landings": landings,
        "date_range": ([min(matched_dates).isoformat(), max(matched_dates).isoformat()] if matched_dates else None),
        "hours": {f: round(v, 1) for f, v in sums.items()},
        "filters": {"through": request.args.get("through"), "from_date": request.args.get("from_date"),
                    "aircraft": aircraft or None, "source": source or None},
    })


@app.route("/api/export/json", methods=["GET"])
@login_required
def export_json():
    """Export logbook as JSON file."""
    logbook = Logbook()
    for entry in storage.get_all_entries(user_id=current_user.id):
        logbook.entries[entry.id] = entry

    json_data = logbook.to_json()
    filename = f"logbook_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        f.write(json_data)

    return send_file(
        filename,
        mimetype="application/json",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/export/csv", methods=["GET"])
@login_required
def export_csv():
    """Export logbook as CSV file."""
    import csv
    from io import StringIO

    entries = storage.get_all_entries(user_id=current_user.id, sort_by_date=True)

    output = StringIO()
    writer = csv.writer(output)

    # Write headers
    writer.writerow(LogbookEntry.get_sheets_headers())

    # Write data rows
    for entry in entries:
        writer.writerow(entry.to_sheets_row())

    output.seek(0)
    filename = f"logbook_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, "w") as f:
        f.write(output.getvalue())

    return send_file(
        filename,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/import/json", methods=["POST"])
@login_required
def import_json():
    """Import logbook entries from JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        content = file.read().decode("utf-8")
        imported = Logbook.from_json(content)

        for entry in imported.entries.values():
            storage.add_entry(entry, user_id=current_user.id)

        return jsonify(
            {"message": f"Imported {len(imported.entries)} entries", "success": True}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============== Google Sheets Backup/Restore ==============


@app.route("/api/sheets/backup", methods=["POST"])
@login_required
def sheets_backup():
    """Backup all logbook entries to Google Sheets."""
    from services.google_sheets import GoogleSheetsService, GoogleSheetsError

    if not current_user.google_refresh_token:
        return jsonify({
            "success": False,
            "error": "Google Sheets access not connected. Please sign in with Google to enable backups.",
            "needs_google_auth": True,
        }), 403

    try:
        service = GoogleSheetsService(current_user.google_refresh_token)
        entries = storage.get_all_entries(user_id=current_user.id, sort_by_date=True)

        sheet_id = service.backup(
            entries=entries,
            sheet_id=current_user.backup_sheet_id or None,
            user_name=current_user.name,
        )

        # Store sheet ID if newly created
        if sheet_id != current_user.backup_sheet_id:
            current_user.backup_sheet_id = sheet_id
            storage.update_user(current_user)

        sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        return jsonify({
            "success": True,
            "message": f"Backed up {len(entries)} entries to Google Sheets",
            "sheet_url": sheet_url,
            "entry_count": len(entries),
        })

    except GoogleSheetsError as e:
        print(f"Sheets backup error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(f"Sheets backup unexpected error: {e}")
        return jsonify({"success": False, "error": f"Backup failed: {str(e)}"}), 500


@app.route("/api/sheets/restore", methods=["POST"])
@login_required
def sheets_restore():
    """Restore logbook entries from a Google Sheet."""
    from services.google_sheets import GoogleSheetsService, GoogleSheetsError

    if not current_user.google_refresh_token:
        return jsonify({
            "success": False,
            "error": "Google Sheets access not connected. Please sign in with Google.",
            "needs_google_auth": True,
        }), 403

    data = request.json or {}
    sheet_url = data.get("sheet_url", "").strip()

    if not sheet_url:
        return jsonify({"success": False, "error": "No Google Sheet URL provided"}), 400

    try:
        sheet_id = GoogleSheetsService.extract_sheet_id(sheet_url)
        service = GoogleSheetsService(current_user.google_refresh_token)
        existing_keys = storage.get_existing_keys(user_id=current_user.id)

        new_entries = service.restore(sheet_id, existing_keys)

        if not new_entries:
            return jsonify({
                "success": True,
                "message": "No new entries found (all entries already exist or sheet is empty)",
                "imported": 0,
            })

        uid = current_user.id
        for entry in new_entries:
            if not entry.source:
                entry.source = "sheets"
            storage.add_entry(entry, user_id=uid)

        return jsonify({
            "success": True,
            "message": f"Restored {len(new_entries)} new entries from Google Sheets",
            "imported": len(new_entries),
        })

    except GoogleSheetsError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Restore failed: {str(e)}"}), 500


# ============== FlightAware Integration ==============


@app.route("/import")
@login_required
def import_page():
    """FlightAware import page."""
    return render_template(
        "import.html",
        tail_number=current_user.default_tail_number or Config.DEFAULT_TAIL_NUMBER,
    )


@app.route("/api/flightaware/search", methods=["POST"])
@login_required
def search_flightaware():
    """Search FlightAware for flights by tail number."""
    from services.flightaware import FlightAwareService

    if not Config.FLIGHTAWARE_API_KEY:
        return jsonify({
            "success": False,
            "error": "FlightAware API key not configured. Set FLIGHTAWARE_API_KEY in .env"
        }), 400

    data = request.json or {}
    tail_number = data.get("tail_number") or current_user.default_tail_number or Config.DEFAULT_TAIL_NUMBER
    start_date = data.get("start_date") or None
    end_date = data.get("end_date") or None
    months_back = data.get("months_back") or 12

    try:
        service = FlightAwareService()
        flights = service.get_flights_as_logbook_entries(
            tail_number=tail_number,
            start_date=start_date,
            end_date=end_date,
            months_back=months_back,
        )

        if flights is None:
            flights = []

        existing_keys = storage.get_existing_keys(user_id=current_user.id)

        for flight in flights:
            route_from = flight.get('route_from') or ''
            route_to = flight.get('route_to') or ''
            date = flight.get('date') or ''
            key = make_entry_key(date, route_from, route_to)
            flight["already_imported"] = key in existing_keys

        return jsonify({
            "success": True,
            "flights": flights,
            "count": len(flights),
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/flightaware/search/stream", methods=["GET"])
@login_required
def search_flightaware_stream():
    """Stream FlightAware search results using OPTIMIZED batch import."""
    import json

    from flask import Response
    from services.flightaware_optimized import OptimizedFlightAwareService

    if not Config.FLIGHTAWARE_API_KEY:
        def error_stream():
            yield f"data: {json.dumps({'error': 'FlightAware API key not configured'})}\n\n"
        return Response(error_stream(), mimetype='text/event-stream')

    uid = current_user.id
    tail_number = request.args.get("tail_number") or current_user.default_tail_number or Config.DEFAULT_TAIL_NUMBER

    most_recent_date = storage.get_most_recent_flight_date(user_id=uid, source="flightaware")
    existing_keys = storage.get_existing_keys(user_id=uid)

    def generate():
        import sys
        service = OptimizedFlightAwareService()
        flight_count = 0
        search_meta = {}
        search_summary = {}
        warnings = []

        try:
            yield ": keepalive\n\n"
            sys.stdout.flush()

            for result in service.stream_flights_ultra_fast(
                tail_number=tail_number,
                most_recent_flight_date=most_recent_date,
                existing_keys=existing_keys,
            ):
                if result.get('_meta'):
                    search_meta = result['_meta']
                    yield f"data: {json.dumps({'meta': search_meta})}\n\n"
                    sys.stdout.flush()
                    continue

                if result.get('_warning'):
                    warnings.append(result['_warning'])
                    yield f"data: {json.dumps({'warning': result['_warning']})}\n\n"
                    sys.stdout.flush()
                    continue

                if result.get('_summary'):
                    search_summary = result['_summary']
                    continue

                if result.get('_keepalive'):
                    yield ": keepalive\n\n"
                    sys.stdout.flush()
                    continue

                if result.get('_batch'):
                    batch = result['_batch']
                    for flight in batch:
                        route_from = flight.get('route_from') or ''
                        route_to = flight.get('route_to') or ''
                        date = flight.get('date') or ''
                        key = make_entry_key(date, route_from, route_to)
                        flight["already_imported"] = key in existing_keys

                        flight_count += 1
                        yield f"data: {json.dumps({'flight': flight})}\n\n"

                    yield ": heartbeat\n\n"
                    sys.stdout.flush()

            done_payload = {
                'done': True,
                'total': flight_count,
                'is_incremental': search_meta.get('is_incremental', False),
                'search_range': f"{search_meta.get('start_date', '?')} to {search_meta.get('end_date', '?')}",
                'api_errors': search_summary.get('api_errors', 0),
                'windows_searched': search_summary.get('windows_attempted', 0),
            }
            if warnings:
                done_payload['warnings'] = warnings
            yield f"data: {json.dumps(done_payload)}\n\n"
        except Exception as e:
            print(f"Error in streaming: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response


@app.route("/api/flightaware/enrich", methods=["POST"])
@login_required
def enrich_flights():
    """Enrich flight entries with airport lookups and calculations."""
    from services.flightaware_optimized import OptimizedFlightAwareService

    data = request.json
    flights = data.get("flights", [])

    if not flights:
        return jsonify({"success": False, "error": "No flight data provided"}), 400

    try:
        service = OptimizedFlightAwareService()
        enriched = service.enrich_batch(flights)
        return jsonify({
            "success": True,
            "flights": enriched
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/approaches/<airport_code>", methods=["GET"])
@login_required
def get_approaches(airport_code):
    """Get instrument approaches for an airport."""
    from services.airport_approaches import get_approaches_for_airport

    approaches = get_approaches_for_airport(airport_code)
    return jsonify({
        "success": True,
        "airport": airport_code.upper(),
        "approaches": approaches,
    })


@app.route("/api/flightaware/import", methods=["POST"])
@login_required
def import_flightaware():
    """Import selected flights from FlightAware into logbook."""
    from services.flightaware_optimized import OptimizedFlightAwareService

    data = request.json
    flights = data.get("flights", [])

    if not flights:
        return jsonify({"success": False, "error": "No flights to import"}), 400

    service = OptimizedFlightAwareService()
    flights = service.enrich_batch(flights)

    uid = current_user.id
    existing_keys = storage.get_existing_keys(user_id=uid)
    imported_count = 0
    skipped_count = 0
    for flight_data in flights:
        flight_data.pop("already_imported", None)
        flight_data.pop("fa_flight_id", None)
        flight_data.pop("duration_estimated", None)
        flight_data.pop("imc_estimated", None)

        entry = LogbookEntry(
            date=flight_data.get("date", ""),
            aircraft_model=flight_data.get("aircraft_model", ""),
            aircraft_ident=flight_data.get("aircraft_ident", "").upper(),
            route_from=normalize_airport(flight_data.get("route_from", "")),
            route_to=normalize_airport(flight_data.get("route_to", "")),
            sel=parse_float(flight_data.get("sel")),
            mel=parse_float(flight_data.get("mel")),
            day=parse_float(flight_data.get("day")),
            night=parse_float(flight_data.get("night")),
            cross_country=parse_float(flight_data.get("cross_country")),
            actual_inst=parse_float(flight_data.get("actual_inst")),
            simulated_inst=parse_float(flight_data.get("simulated_inst")),
            num_inst_app=parse_int(flight_data.get("num_inst_app")),
            landings_day=parse_int(flight_data.get("landings_day")),
            landings_night=parse_int(flight_data.get("landings_night")),
            pic=parse_float(flight_data.get("pic")),
            sic=parse_float(flight_data.get("sic")),
            dual_recd=parse_float(flight_data.get("dual_recd")),
            dual_given=parse_float(flight_data.get("dual_given")),
            solo=parse_float(flight_data.get("solo")),
            sim=parse_float(flight_data.get("sim")),
            total_duration=parse_float(flight_data.get("total_duration")),
            remarks=flight_data.get("remarks", ""),
            reviewed=False,
            source="flightaware",
        )

        key = make_entry_key(entry.date, entry.route_from, entry.route_to)
        existing = existing_keys.get(key)
        if existing:
            skipped_count += 1
            continue

        storage.add_entry(entry, user_id=uid)
        existing_keys[key] = {"id": entry.id, "source": "flightaware"}
        imported_count += 1

    return jsonify({
        "success": True,
        "message": f"Imported {imported_count} flight(s)" + (f", skipped {skipped_count} duplicate(s)" if skipped_count else ""),
        "imported": imported_count,
        "skipped": skipped_count,
    })


@app.route("/scan")
@login_required
def scan_page():
    """Render logbook scanning interface."""
    return render_template("scan.html")


@app.route("/api/logbook/scan/upload", methods=["POST"])
@login_required
def upload_scan():
    """Accept image upload, extract flight entries using Gemini AI."""
    import uuid
    from services.ocr_service import LogbookOCRService

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    from services.ocr_service import OCRAuthError, OCRError
    temp_path = None
    try:
        temp_path = f"/tmp/logbook_{uuid.uuid4()}.jpg"
        file.save(temp_path)

        t0 = time.time()

        known_idents, known_models, known_airports = storage.get_known_values(user_id=current_user.id)

        t1 = time.time()

        print(f"Starting Gemini extraction for: {temp_path}")
        # Per-logbook conventions (year recording + checksum) vary book to book. The client
        # collects these once when the user starts scanning a logbook and passes them here.
        year_guidance = (request.form.get("year_guidance") or "").strip() or None
        checksum_guidance = (request.form.get("checksum_guidance") or "").strip() or None
        ocr_service = LogbookOCRService()
        entries, expected_rows, actual_rows = ocr_service.extract_flights_with_gemini(
            temp_path, known_idents=known_idents, known_models=known_models,
            known_airports=known_airports, year_guidance=year_guidance
        )

        t2 = time.time()
        print(f"Scan timing: db_queries={t1-t0:.2f}s, gemini+postprocess={t2-t1:.2f}s, total={t2-t0:.2f}s")

        if not entries:
            return jsonify({
                "success": False,
                "error": "No flight entries detected. Please ensure the image shows a logbook page with flight data."
            }), 400

        print(f"Successfully extracted {actual_rows} of {expected_rows} entries")

        # Independent page-total checksum (separate read; flags mismatches, never edits rows).
        with open(temp_path, 'rb') as _f:
            _img_bytes = _f.read()
        printed_total = ocr_service.read_printed_page_total(_img_bytes, checksum_guidance=checksum_guidance)
        checksum = LogbookOCRService.checksum_entries(entries, printed_total)
        if not checksum["reconciled"]:
            print(f"CHECKSUM FLAG: rows={checksum['row_sum']} vs printed={printed_total} "
                  f"diff={checksum['difference']} implausible={len(checksum['implausible_rows'])}")

        return jsonify({
            "success": True,
            "entries": entries,
            "count": len(entries),
            "expected_rows": expected_rows,
            "checksum": checksum,
        })

    except OCRAuthError as e:
        # Configuration/auth problem — surface the real reason (don't mask as "no entries").
        return jsonify({"success": False, "error": f"Scanning is not available: {e}",
                        "kind": "auth"}), 502
    except OCRError as e:
        return jsonify({"success": False, "error": f"Scan failed: {e}", "kind": "ocr"}), 502
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/api/logbook/scan/import", methods=["POST"])
@login_required
def import_scanned():
    """Import scanned logbook entries into database."""
    data = request.json
    entries = data.get("entries", [])

    if not entries:
        return jsonify({"success": False, "error": "No entries to import"}), 400

    uid = current_user.id
    existing_keys = storage.get_existing_keys(user_id=uid)
    entry_ids = []
    updated_count = 0
    for entry_data in entries:
        entry_data.pop("_raw", None)

        entry = LogbookEntry(
            date=entry_data.get("date", ""),
            aircraft_model=entry_data.get("aircraft_model", ""),
            aircraft_ident=entry_data.get("aircraft_ident", "").upper(),
            route_from=normalize_airport(entry_data.get("route_from", "")),
            route_to=normalize_airport(entry_data.get("route_to", "")),
            route_via=normalize_route_via(entry_data.get("route_via", "")),
            sel=parse_float(entry_data.get("sel")),
            mel=parse_float(entry_data.get("mel")),
            day=parse_float(entry_data.get("day")),
            night=parse_float(entry_data.get("night")),
            cross_country=parse_float(entry_data.get("cross_country")),
            actual_inst=parse_float(entry_data.get("actual_inst")),
            simulated_inst=parse_float(entry_data.get("simulated_inst")),
            num_inst_app=parse_int(entry_data.get("num_inst_app")),
            landings_day=parse_int(entry_data.get("landings_day")),
            landings_night=parse_int(entry_data.get("landings_night")),
            pic=parse_float(entry_data.get("pic")),
            sic=parse_float(entry_data.get("sic")),
            dual_recd=parse_float(entry_data.get("dual_recd")),
            dual_given=parse_float(entry_data.get("dual_given")),
            solo=parse_float(entry_data.get("solo")),
            sim=parse_float(entry_data.get("sim")),
            total_duration=parse_float(entry_data.get("total_duration")),
            remarks=entry_data.get("remarks", ""),
            reviewed=False,
            source="scan",
        )

        key = make_entry_key(entry.date, entry.route_from, entry.route_to)
        existing = existing_keys.get(key)
        if existing:
            # Merge: scan values win on conflicts, otherwise combine
            old = storage.get_entry(existing["id"], user_id=uid)
            if old:
                entry.id = old.id
                # Numeric fields: prefer scan value if non-zero, else keep existing
                for fld in ('sel', 'mel', 'day', 'night', 'cross_country',
                            'actual_inst', 'simulated_inst', 'pic', 'sic',
                            'dual_recd', 'dual_given', 'solo', 'sim',
                            'total_duration'):
                    scan_val = getattr(entry, fld)
                    old_val = getattr(old, fld)
                    setattr(entry, fld, scan_val if scan_val else old_val)
                for fld in ('num_inst_app', 'landings_day', 'landings_night'):
                    scan_val = getattr(entry, fld)
                    old_val = getattr(old, fld)
                    setattr(entry, fld, scan_val if scan_val else old_val)
                # String fields: prefer scan if non-empty, else keep existing
                for fld in ('aircraft_model', 'aircraft_ident', 'route_via', 'remarks'):
                    scan_val = getattr(entry, fld)
                    old_val = getattr(old, fld)
                    setattr(entry, fld, scan_val if scan_val else old_val)
                # Keep reviewed/locked status from existing entry
                entry.reviewed = old.reviewed
                entry.locked = old.locked
                entry.created_at = old.created_at
                # Source: mark as both if different
                if old.source and old.source != "scan":
                    entry.source = f"{old.source}+scan"
                # Keep FlightAware's estimated flag only if scan didn't provide duration
                if not entry_data.get("total_duration"):
                    entry.duration_estimated = old.duration_estimated
                else:
                    entry.duration_estimated = False
            storage.update_entry(entry, user_id=uid)
            entry_ids.append(entry.id)
            updated_count += 1
        else:
            entry_id = storage.add_entry(entry, user_id=uid)
            entry_ids.append(entry_id)
            existing_keys[key] = {"id": entry_id, "source": "scan"}

    new_count = len(entry_ids) - updated_count
    parts = []
    if new_count:
        parts.append(f"{new_count} new")
    if updated_count:
        parts.append(f"{updated_count} updated")
    msg = f"Imported {' and '.join(parts)} flight(s) from scanned logbook"

    return jsonify({
        "success": True,
        "imported": len(entry_ids),
        "new": new_count,
        "updated": updated_count,
        "entry_ids": entry_ids,
        "message": msg,
    })


if __name__ == "__main__":
    print(f"Starting Pilot Logbook on http://{Config.HOST}:{Config.PORT}")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
