"""Focus Flow application server (Figure 3.1: API + business logic).

Run:  python app.py        then open http://localhost:5000
"""
import csv
import io
import json
import os
import re
import secrets
import time
from functools import wraps

from flask import Flask, Response, g, jsonify, request, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash

import db
import forms
import ml
import services as svc
import stats

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")


def _secret():
    path = os.path.join(os.path.dirname(db.DB_PATH), ".secret_key")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    return open(path).read().strip()


app.secret_key = os.environ.get("FOCUSFLOW_SECRET") or _secret()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.environ.get("FOCUSFLOW_HTTPS") == "1",
                  PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30, MAX_CONTENT_LENGTH=256 * 1024)

MAX_SESSION_SECONDS = 8 * 3600       # Section 3.2: sessions longer than 8 hours are errors
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ------------------------------------------------------------------------------------------ plumbing
def get_conn():
    if "conn" not in g:
        g.conn = db.connect()
    return g.conn


@app.teardown_appcontext
def close_conn(_):
    c = g.pop("conn", None)
    if c is not None:
        c.close()


@app.before_request
def start_timer():
    g.t0 = time.time()


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Content-Security-Policy"] = ("default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                               "img-src 'self' data:; connect-src 'self'")
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Response-Time-ms"] = str(int((time.time() - g.t0) * 1000))   # NFR2 / Table 3.6
    return resp


def err(msg, code=400):
    return jsonify({"error": msg}), code


@app.errorhandler(404)
def nf(_):
    if request.path.startswith("/api/"):
        return err("Not found", 404)
    return send_from_directory(app.static_folder, "index.html")


@app.errorhandler(Exception)
def boom(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return err(e.description, e.code)
    app.logger.exception("Unhandled error")
    try:
        conn = get_conn()
        conn.execute("INSERT INTO events(user_id,type,payload,created_at) "
                     "SELECT ?, 'server_error', ?, ?", (session.get("uid") or 0, json.dumps({"path": request.path}), svc.now()))
        conn.commit()
    except Exception:
        pass
    return err("Something went wrong on the server.", 500)


def current_user():
    uid = session.get("uid")
    if not uid:
        return None
    return db.row(get_conn(), "SELECT * FROM users WHERE id=?", (uid,))


def login_required(role=None):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            u = current_user()
            if not u:
                return err("Please sign in.", 401)
            if role and u["role"] != role:
                return err("Not allowed.", 403)
            g.user = u
            return fn(*a, **kw)
        return wrapper
    return deco


def public_user(u):
    return {k: u[k] for k in ("id", "email", "name", "role", "code", "consent", "programme", "level", "gender")}


def own_task(task_id):
    return db.row(get_conn(), "SELECT * FROM tasks WHERE id=? AND user_id=?", (task_id, g.user["id"]))


def parse_int(v, lo, hi, field):
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"{field} must be between {lo} and {hi}.")
    return v


# ------------------------------------------------------------------------------------------------ FR1
@app.post("/api/register")
def register():
    d = request.get_json(silent=True) or {}
    email = (d.get("email") or "").strip().lower()
    name = (d.get("name") or "").strip()[:80]
    pw = d.get("password") or ""
    if not EMAIL_RE.match(email):
        return err("Enter a valid email address.")
    if not name:
        return err("Enter your name.")
    if len(pw) < 8:
        return err("Password must be at least 8 characters.")
    conn = get_conn()
    if db.row(conn, "SELECT 1 FROM users WHERE email=?", (email,)):
        return err("An account with that email already exists.", 409)
    code = "P" + secrets.token_hex(3).upper()
    cur = conn.execute(
        "INSERT INTO users(email,name,pw_hash,role,code,consent,programme,level,gender,tz_offset_min,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (email, name, generate_password_hash(pw), "student", code, 1 if d.get("consent") else 0,
         (d.get("programme") or "")[:80] or None, (d.get("level") or "")[:20] or None,
         (d.get("gender") or "")[:20] or None, int(d.get("tz_offset_min") or 0), svc.now()))
    conn.commit()
    session.permanent = True
    session["uid"] = cur.lastrowid
    u = db.row(conn, "SELECT * FROM users WHERE id=?", (cur.lastrowid,))
    svc.log_event(conn, u, "register", {})
    conn.commit()
    return jsonify({"user": public_user(u)})


@app.post("/api/login")
def login():
    d = request.get_json(silent=True) or {}
    conn = get_conn()
    u = db.row(conn, "SELECT * FROM users WHERE email=?", ((d.get("email") or "").strip().lower(),))
    if not u or not check_password_hash(u["pw_hash"], d.get("password") or ""):
        return err("Wrong email or password.", 401)
    session.permanent = True
    session["uid"] = u["id"]
    if d.get("tz_offset_min") is not None:
        conn.execute("UPDATE users SET tz_offset_min=? WHERE id=?", (int(d["tz_offset_min"]), u["id"]))
        u["tz_offset_min"] = int(d["tz_offset_min"])
    svc.log_event(conn, u, "login", {})
    conn.commit()
    return jsonify({"user": public_user(u)})


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/me")
def me():
    u = current_user()
    if not u:
        return jsonify({"user": None})
    conn = get_conn()
    done = {r["kind"] for r in db.rows(conn, "SELECT kind FROM surveys WHERE user_id=?", (u["id"],))}
    first_task = db.row(conn, "SELECT MIN(created_at) m FROM tasks WHERE user_id=?", (u["id"],))["m"]
    return jsonify({"user": public_user(u), "surveys_done": sorted(done),
                    "study_week": None if not first_task else int((svc.now() - first_task) / (7 * 86400)) + 1})


@app.post("/api/profile")
@login_required()
def update_profile():
    d = request.get_json(silent=True) or {}
    conn = get_conn()
    name = (d.get("name") or g.user["name"]).strip()[:80]
    conn.execute("UPDATE users SET name=?, programme=?, level=?, gender=? WHERE id=?",
                 (name, (d.get("programme") or "")[:80] or None, (d.get("level") or "")[:20] or None,
                  (d.get("gender") or "")[:20] or None, g.user["id"]))
    if "consent" in d:
        conn.execute("UPDATE users SET consent=? WHERE id=?", (1 if d["consent"] else 0, g.user["id"]))
    if d.get("password"):
        if len(d["password"]) < 8:
            return err("Password must be at least 8 characters.")
        conn.execute("UPDATE users SET pw_hash=? WHERE id=?", (generate_password_hash(d["password"]), g.user["id"]))
    conn.commit()
    return jsonify({"user": public_user(db.row(conn, "SELECT * FROM users WHERE id=?", (g.user["id"],)))})


@app.delete("/api/account")
@login_required("student")
def delete_account():
    """Right to withdraw (Section 3.1): removes the account and every record tied to it."""
    conn = get_conn()
    conn.execute("DELETE FROM users WHERE id=?", (g.user["id"],))
    conn.commit()
    session.clear()
    return jsonify({"ok": True})


# --------------------------------------------------------------------------------------- surveys
@app.get("/api/forms")
def get_forms():
    return jsonify(forms.public_forms())


def scrub(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", text or "")
    return re.sub(r"\+?\d[\d\s-]{7,}\d", "[number]", text)[:1000]


@app.post("/api/survey")
@login_required("student")
def submit_survey():
    d = request.get_json(silent=True) or {}
    kind, answers = d.get("kind"), d.get("answers")
    try:
        if kind in ("pre", "post"):
            score = forms.score_procrastination(answers)
        elif kind == "sus":
            score = forms.score_sus(answers)
        elif kind == "needs":
            score = None
        else:
            return err("Unknown questionnaire.")
    except ValueError as e:
        return err(str(e))
    payload = {"answers": answers}
    if kind == "sus":
        payload["feedback"] = scrub(d.get("feedback"))
    conn = get_conn()
    conn.execute("INSERT INTO surveys(user_id,kind,answers,score,created_at) VALUES(?,?,?,?,?) "
                 "ON CONFLICT(user_id,kind) DO UPDATE SET answers=excluded.answers, score=excluded.score, "
                 "created_at=excluded.created_at",
                 (g.user["id"], kind, json.dumps(payload), score, svc.now()))
    svc.log_event(conn, g.user, "survey_" + kind, {"score": score})
    conn.commit()
    return jsonify({"ok": True, "score": score})


# ------------------------------------------------------------------------------------- FR2 / FR3
def read_task_payload(d):
    title = (d.get("title") or "").strip()[:140]
    if not title:
        raise ValueError("Give the task a title.")
    ttype = d.get("task_type") if d.get("task_type") in svc.TASK_TYPES else "other"
    minutes = parse_int(d.get("planned_minutes"), 10, 2400, "Estimated effort (minutes)")
    try:
        deadline = int(d.get("deadline"))
    except (TypeError, ValueError):
        raise ValueError("Choose a deadline.")
    created = svc.now()
    if deadline <= created + 1800:
        raise ValueError("The deadline must be at least 30 minutes from now.")
    if deadline > created + 400 * 86400:
        raise ValueError("That deadline is too far away.")
    return title, ttype, minutes, deadline, created


@app.post("/api/tasks/preview")
@login_required("student")
def preview_task():
    try:
        title, ttype, minutes, deadline, created = read_task_payload(request.get_json(silent=True) or {})
    except ValueError as e:
        return err(str(e))
    return jsonify({"steps": svc.breakdown(title, ttype, minutes, created, deadline)})


@app.post("/api/tasks")
@login_required("student")
def create_task():
    d = request.get_json(silent=True) or {}
    try:
        title, ttype, minutes, deadline, created = read_task_payload(d)
    except ValueError as e:
        return err(str(e))
    steps = svc.breakdown(title, ttype, minutes, created, deadline)
    # student may have edited the proposed steps in the UI: accept titles/minutes if consistent
    edited = d.get("steps")
    if isinstance(edited, list) and 1 <= len(edited) <= 14 and len(edited) == len(steps):
        for s, e in zip(steps, edited):
            t = (e.get("title") or "").strip()[:140]
            if t:
                s["title"] = t
    conn = get_conn()
    feat = svc.capture_features(conn, g.user, ttype, minutes, created, deadline, steps)
    cur = conn.execute("INSERT INTO tasks(user_id,title,task_type,created_at,deadline,planned_minutes,features,is_demo) "
                       "VALUES(?,?,?,?,?,?,?,?)",
                       (g.user["id"], title, ttype, created, deadline, minutes, json.dumps(feat), g.user["is_demo"]))
    svc.save_steps(conn, cur.lastrowid, steps)
    risk, src = ml.predict_risk(feat)
    svc.log_event(conn, g.user, "task_created", {"task_id": cur.lastrowid, "type": ttype, "planned": minutes,
                                                 "hours_available": feat["hours_available"], "n_steps": len(steps),
                                                 "risk": round(risk, 3)})
    conn.commit()
    return jsonify({"task": svc.task_view(conn, db.row(conn, "SELECT * FROM tasks WHERE id=?", (cur.lastrowid,)))})


@app.get("/api/tasks")
@login_required("student")
def list_tasks():
    conn = get_conn()
    rows = db.rows(conn, "SELECT * FROM tasks WHERE user_id=? ORDER BY status='done', deadline", (g.user["id"],))
    return jsonify({"tasks": [svc.task_view(conn, t) for t in rows], "model": ml.model_info().get("best_model")})


@app.post("/api/tasks/<int:task_id>/complete")
@login_required("student")
def complete_task(task_id):
    t = own_task(task_id)
    if not t:
        return err("Task not found.", 404)
    conn = get_conn()
    ts = svc.now()
    conn.execute("UPDATE tasks SET status='done', completed_at=? WHERE id=?", (ts, task_id))
    conn.execute("UPDATE steps SET done=1 WHERE task_id=?", (task_id,))
    svc.log_event(conn, g.user, "task_completed", {"task_id": task_id, "on_time": ts <= t["deadline"]})
    conn.commit()
    return jsonify({"ok": True, "on_time": ts <= t["deadline"]})


@app.post("/api/tasks/<int:task_id>/reopen")
@login_required("student")
def reopen_task(task_id):
    if not own_task(task_id):
        return err("Task not found.", 404)
    conn = get_conn()
    conn.execute("UPDATE tasks SET status='open', completed_at=NULL WHERE id=?", (task_id,))
    conn.commit()
    return jsonify({"ok": True})


@app.post("/api/tasks/<int:task_id>/postpone")
@login_required("student")
def postpone_task(task_id):
    """Reschedule the remaining steps later (counted as a postponement, Table 3.2)."""
    t = own_task(task_id)
    if not t:
        return err("Task not found.", 404)
    conn = get_conn()
    ts = svc.now()
    pending = db.rows(conn, "SELECT id,scheduled_for FROM steps WHERE task_id=? AND done=0", (task_id,))
    latest = t["deadline"] - 1800
    for s in pending:
        conn.execute("UPDATE steps SET scheduled_for=? WHERE id=?", (min(max(s["scheduled_for"], ts) + 86400, latest), s["id"]))
    conn.execute("UPDATE tasks SET postponements=postponements+1 WHERE id=?", (task_id,))
    svc.log_event(conn, g.user, "task_postponed", {"task_id": task_id})
    conn.commit()
    return jsonify({"task": svc.task_view(conn, db.row(conn, "SELECT * FROM tasks WHERE id=?", (task_id,)))})


@app.delete("/api/tasks/<int:task_id>")
@login_required("student")
def delete_task(task_id):
    if not own_task(task_id):
        return err("Task not found.", 404)
    conn = get_conn()
    conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    conn.commit()
    return jsonify({"ok": True})


@app.post("/api/steps/<int:step_id>/toggle")
@login_required("student")
def toggle_step(step_id):
    conn = get_conn()
    s = db.row(conn, "SELECT s.* FROM steps s JOIN tasks t ON t.id=s.task_id WHERE s.id=? AND t.user_id=?",
               (step_id, g.user["id"]))
    if not s:
        return err("Step not found.", 404)
    conn.execute("UPDATE steps SET done=? WHERE id=?", (0 if s["done"] else 1, step_id))
    left = db.row(conn, "SELECT COUNT(*) n FROM steps WHERE task_id=? AND done=0", (s["task_id"],))["n"]
    svc.log_event(conn, g.user, "step_toggled", {"task_id": s["task_id"], "done": not s["done"]})
    conn.commit()
    return jsonify({"ok": True, "done": not s["done"], "all_done": left == 0})


# --------------------------------------------------------------------------------------------- FR4
@app.post("/api/sessions/start")
@login_required("student")
def start_session():
    d = request.get_json(silent=True) or {}
    conn = get_conn()
    task_id = d.get("task_id")
    if task_id is not None and not own_task(int(task_id)):
        return err("Task not found.", 404)
    try:
        minutes = parse_int(d.get("planned_minutes", 25), 5, 120, "Session length")
    except ValueError as e:
        return err(str(e))
    # close any session left running (e.g. browser closed): count nothing beyond its planned time
    for r in db.rows(conn, "SELECT * FROM sessions WHERE user_id=? AND status='running'", (g.user["id"],)):
        conn.execute("UPDATE sessions SET status='abandoned', ended_at=? WHERE id=?", (svc.now(), r["id"]))
    ts = svc.now()
    cur = conn.execute("INSERT INTO sessions(user_id,task_id,started_at,planned_minutes,is_demo) VALUES(?,?,?,?,?)",
                       (g.user["id"], task_id, ts, minutes, g.user["is_demo"]))
    if task_id:
        conn.execute("UPDATE tasks SET first_start_at=COALESCE(first_start_at,?) WHERE id=?", (ts, task_id))
    svc.log_event(conn, g.user, "session_start", {"session_id": cur.lastrowid, "task_id": task_id, "planned": minutes})
    conn.commit()
    return jsonify({"session_id": cur.lastrowid, "started_at": ts})


@app.post("/api/sessions/<int:sid>/interrupt")
@login_required("student")
def interrupt_session(sid):
    conn = get_conn()
    cur = conn.execute("UPDATE sessions SET interruptions=interruptions+1 WHERE id=? AND user_id=? AND status='running'",
                       (sid, g.user["id"]))
    svc.log_event(conn, g.user, "session_interrupt", {"session_id": sid})
    conn.commit()
    return jsonify({"ok": cur.rowcount == 1})


@app.post("/api/sessions/<int:sid>/finish")
@login_required("student")
def finish_session(sid):
    d = request.get_json(silent=True) or {}
    conn = get_conn()
    s = db.row(conn, "SELECT * FROM sessions WHERE id=? AND user_id=?", (sid, g.user["id"]))
    if not s:
        return err("Session not found.", 404)
    ts = svc.now()
    elapsed = max(0, ts - s["started_at"])
    focus = max(0, min(int(d.get("focus_seconds") or 0), elapsed + 5, MAX_SESSION_SECONDS))   # Section 3.2 sanity limits
    status = "finished" if d.get("completed", True) else "abandoned"
    conn.execute("UPDATE sessions SET ended_at=?, focus_seconds=?, status=? WHERE id=?", (ts, focus, status, sid))
    step_done = False
    if d.get("step_id") and d.get("step_completed"):
        st = db.row(conn, "SELECT s.id FROM steps s JOIN tasks t ON t.id=s.task_id WHERE s.id=? AND t.user_id=?",
                    (int(d["step_id"]), g.user["id"]))
        if st:
            conn.execute("UPDATE steps SET done=1 WHERE id=?", (st["id"],))
            step_done = True
    svc.log_event(conn, g.user, "session_end", {"session_id": sid, "focus_seconds": focus, "status": status})
    conn.commit()
    return jsonify({"ok": True, "focus_seconds": focus, "step_done": step_done})


# --------------------------------------------------------------------------------- FR5 / FR6
@app.get("/api/nudges")
@login_required("student")
def nudges():
    conn = get_conn()
    svc.generate_nudges(conn, g.user)
    conn.commit()
    rows = db.rows(conn, "SELECT id,task_id,kind,message,risk,created_at,seen FROM nudges WHERE user_id=? "
                         "ORDER BY created_at DESC LIMIT 20", (g.user["id"],))
    return jsonify({"nudges": rows, "unseen": sum(1 for r in rows if not r["seen"])})


@app.post("/api/nudges/<int:nid>/seen")
@login_required("student")
def nudge_seen(nid):
    conn = get_conn()
    conn.execute("UPDATE nudges SET seen=1 WHERE id=? AND user_id=?", (nid, g.user["id"]))
    svc.log_event(conn, g.user, "nudge_seen", {"nudge_id": nid})
    conn.commit()
    return jsonify({"ok": True})


@app.get("/api/dashboard")
@login_required("student")
def dashboard():
    return jsonify(svc.dashboard(get_conn(), g.user))


@app.post("/api/event")
@login_required("student")
def client_event():
    d = request.get_json(silent=True) or {}
    etype = (d.get("type") or "")[:40]
    if etype in {"view_dashboard", "view_planner", "view_focus", "view_study", "pwa_install"}:
        svc.log_event(get_conn(), g.user, etype, {})
        get_conn().commit()
    return jsonify({"ok": True})


# --------------------------------------------------------------------------------- researcher (FR7)
def csv_response(name, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


def demo_flag():
    return request.args.get("demo", "1") != "0"


@app.get("/api/research/overview")
@login_required("researcher")
def research_overview():
    conn = get_conn()
    c = lambda sql: db.row(conn, sql)["n"]
    return jsonify({
        "students": c("SELECT COUNT(*) n FROM users WHERE role='student' AND is_demo=0"),
        "consenting": c("SELECT COUNT(*) n FROM users WHERE role='student' AND is_demo=0 AND consent=1"),
        "demo_students": c("SELECT COUNT(*) n FROM users WHERE is_demo=1"),
        "tasks": c("SELECT COUNT(*) n FROM tasks WHERE is_demo=0"),
        "sessions": c("SELECT COUNT(*) n FROM sessions WHERE is_demo=0"),
        "events": c("SELECT COUNT(*) n FROM events e JOIN users u ON u.id=e.user_id WHERE u.is_demo=0"),
        "model": ml.model_info(), "min_tasks": ml.MIN_TASKS, "min_students": ml.MIN_STUDENTS})


@app.get("/api/research/report")
@login_required("researcher")
def research_report():
    return jsonify(stats.build_report(include_demo=demo_flag()))


@app.get("/api/research/report.md")
@login_required("researcher")
def research_report_md():
    return Response(stats.report_markdown(stats.build_report(include_demo=demo_flag())), mimetype="text/markdown",
                    headers={"Content-Disposition": "attachment; filename=focusflow_results.md"})


@app.post("/api/research/train")
@login_required("researcher")
def research_train():
    d = request.get_json(silent=True) or {}
    try:
        meta = ml.train(include_demo=bool(d.get("include_demo", True)))
    except ValueError as e:
        return err(str(e))
    return jsonify({"model": meta})


@app.get("/api/research/export/tasks.csv")
@login_required("researcher")
def export_tasks():
    """Table 3.2 data dictionary. Participants are identified only by their pseudonymous code."""
    conn = get_conn()
    demo = "" if demo_flag() else " AND t.is_demo=0"
    ts = svc.now()
    out = []
    for t in db.rows(conn, "SELECT t.*, u.code FROM tasks t JOIN users u ON u.id=t.user_id "
                           f"WHERE u.consent=1{demo} ORDER BY t.created_at"):
        f = json.loads(t["features"]) if t["features"] else {}
        focus = db.row(conn, "SELECT COALESCE(SUM(focus_seconds),0) s, COALESCE(SUM(interruptions),0) i FROM sessions "
                             "WHERE task_id=?", (t["id"],))
        out.append([t["code"], t["id"], t["task_type"], t["created_at"], t["deadline"], t["first_start_at"] or "",
                    round((t["first_start_at"] - t["created_at"]) / 3600, 2) if t["first_start_at"] else "",
                    t["planned_minutes"], round(focus["s"] / 60, 1), focus["i"], t["postponements"],
                    (1 if t["completed_at"] <= t["deadline"] else 0) if t["completed_at"] else 0,
                    ml.label_task(t, ts) if ml.label_task(t, ts) is not None else "",
                    t["is_demo"]] + [f.get(k, "") for k in ml.FEATURES])
    return csv_response("focusflow_tasks.csv",
                        ["student_id", "task_id", "task_type", "created_at", "deadline", "first_start_at", "start_delay_h",
                         "planned_duration", "focus_minutes", "interruptions", "postponements", "completed_on_time",
                         "procrastinated", "is_demo"] + ml.FEATURES, out)


@app.get("/api/research/export/events.csv")
@login_required("researcher")
def export_events():
    conn = get_conn()
    demo = "" if demo_flag() else " AND u.is_demo=0"
    rows = db.rows(conn, "SELECT u.code, e.type, e.payload, e.created_at FROM events e JOIN users u ON u.id=e.user_id "
                         f"WHERE u.consent=1{demo} ORDER BY e.created_at")
    return csv_response("focusflow_events.csv", ["student_id", "event_type", "payload", "created_at"],
                        [[r["code"], r["type"], r["payload"], r["created_at"]] for r in rows])


@app.get("/api/research/export/surveys.csv")
@login_required("researcher")
def export_surveys():
    conn = get_conn()
    demo = "" if demo_flag() else " AND u.is_demo=0"
    rows = db.rows(conn, "SELECT u.code, u.gender, u.level, u.programme, s.kind, s.score, s.created_at "
                         "FROM surveys s JOIN users u ON u.id=s.user_id "
                         f"WHERE u.consent=1 AND s.kind!='needs'{demo} ORDER BY u.code")
    return csv_response("focusflow_surveys.csv", ["student_id", "gender", "level", "programme", "questionnaire", "score",
                                                   "submitted_at"],
                        [[r["code"], r["gender"], r["level"], r["programme"], r["kind"], r["score"], r["created_at"]]
                         for r in rows])


@app.post("/api/research/demo")
@login_required("researcher")
def research_demo():
    import seed
    d = request.get_json(silent=True) or {}
    if d.get("clear"):
        seed.clear_demo()
        return jsonify({"ok": True, "cleared": True})
    return jsonify({"ok": True, **seed.seed_demo(n_students=40)})


# --------------------------------------------------------------------------------------- static
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


def bootstrap():
    db.init()
    conn = db.connect()
    # accounts for the demo and for the researcher (change these passwords before any real deployment)
    for email, name, role, pw in (("researcher@focusflow.app", "Researcher", "researcher",
                                   os.environ.get("FOCUSFLOW_RESEARCHER_PW", "researcher123")),
                                  ("demo@focusflow.app", "Demo Student", "student", "demo12345")):
        if not db.row(conn, "SELECT 1 FROM users WHERE email=?", (email,)):
            conn.execute("INSERT INTO users(email,name,pw_hash,role,code,consent,programme,level,gender,created_at,is_demo)"
                         " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                         (email, name, generate_password_hash(pw), role, "R000" if role == "researcher" else "DEMO01",
                          1 if role == "student" else 0, "Computer Science", "Level 300", "Prefer not to say", svc.now(),
                          1 if role == "student" else 0))
    conn.commit()
    conn.close()


bootstrap()
import seed  # noqa: E402
seed.ensure_demo_account()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 5000)), debug=False)
