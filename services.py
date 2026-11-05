"""Business logic: assignment breakdown, feature capture, notification service, dashboard, event log."""
import json
import math
import time
from datetime import datetime, timedelta, timezone

import db
import ml

TASK_TYPES = ["assignment", "reading", "revision", "other"]

STEP_TEMPLATES = {
    "assignment": ["Open the brief and list exactly what is required", "Outline your answer / structure",
                   "Gather notes and sources", "Draft the main body", "Finish the draft",
                   "Review against the brief and submit"],
    "reading": ["Skim headings and set 3 questions to answer", "Read the first part and note key points",
                "Read the next part and note key points", "Read the final part", "Summarise in your own words"],
    "revision": ["List the topics to cover", "Review notes on the first topics", "Practise questions",
                 "Review notes on the next topics", "Do a timed past question", "Fix weak spots"],
    "other": ["Define the first small action", "Work through the main part", "Finish and check"],
}


def now():
    return int(time.time())


def local_dt(ts, tz_offset_min):
    return datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(minutes=tz_offset_min)


# --------------------------------------------------------------------------------------- events (FR7)
def log_event(conn, user, etype, payload=None):
    """Record an interaction event only if the participant consented to research data collection."""
    if not user or not user.get("consent"):
        return
    conn.execute("INSERT INTO events(user_id,type,payload,created_at) VALUES(?,?,?,?)",
                 (user["id"], etype, json.dumps(payload or {}), now()))


# ----------------------------------------------------------------------------------- breakdown (FR3)
def breakdown(title, task_type, planned_minutes, created, deadline):
    """Split a task into small scheduled steps with implementation-intention wording.

    * the first step is deliberately tiny (<=10 min) so that starting is easy (Fogg: ability)
    * remaining effort is split into 25-45 minute blocks (time-boxing)
    * steps are spread across the available window, leaving a 15% buffer before the deadline
    """
    planned = max(int(planned_minutes), 10)
    first = min(10, planned)
    rest = planned - first
    block = 25 if planned <= 150 else 45
    n_rest = max(1, math.ceil(rest / block)) if rest > 0 else 0
    n_rest = min(n_rest, 11)
    templ = STEP_TEMPLATES.get(task_type, STEP_TEMPLATES["other"])

    titles = [templ[0]]
    if n_rest:
        mids = templ[1:-1] if len(templ) > 2 else []
        for i in range(n_rest - 1):
            titles.append(mids[i] if i < len(mids) else f"Work block {i + 1}")
        titles.append(templ[-1])
    minutes = [first]
    if n_rest:
        base, extra = divmod(rest, n_rest)
        minutes += [base + (1 if i < extra else 0) for i in range(n_rest)]

    window = max(deadline - created, 3600)
    usable = window * 0.85
    n = len(titles)
    steps = []
    for i, (t, m) in enumerate(zip(titles, minutes)):
        # first step soon after creation, later steps spread evenly
        frac = (i / n) if n > 1 else 0
        when = created + int(min(usable * frac + 900 * (i == 0), usable))
        steps.append({"position": i, "title": t, "minutes": int(m), "scheduled_for": int(when)})
    return steps


def save_steps(conn, task_id, steps):
    conn.execute("DELETE FROM steps WHERE task_id=?", (task_id,))
    conn.executemany("INSERT INTO steps(task_id,position,title,minutes,scheduled_for) VALUES(?,?,?,?,?)",
                     [(task_id, s["position"], s["title"], s["minutes"], s["scheduled_for"]) for s in steps])


# ------------------------------------------------------------------------------------- features (3.4)
def capture_features(conn, user, task_type, planned_minutes, created, deadline, steps):
    """Snapshot of Table 3.3 features, using only information available at task creation."""
    uid = user["id"]
    ldt = local_dt(created, user["tz_offset_min"])
    week_ago = created - 7 * 86400
    focus = db.row(conn, "SELECT COALESCE(SUM(focus_seconds),0) s FROM sessions WHERE user_id=? AND started_at>=? "
                         "AND started_at<?", (uid, week_ago, created))["s"]
    sess = db.row(conn, "SELECT COUNT(*) n, COALESCE(SUM(interruptions),0) i FROM sessions WHERE user_id=? "
                        "AND started_at<?", (uid, created))
    past = db.rows(conn, "SELECT * FROM tasks WHERE user_id=? AND created_at<?", (uid, created))
    labelled = [ml.label_task(t, created) for t in past]
    labelled = [x for x in labelled if x is not None]
    concurrent = db.row(conn, "SELECT COUNT(*) n FROM tasks WHERE user_id=? AND created_at<? AND deadline>? "
                              "AND (completed_at IS NULL OR completed_at>?)", (uid, created, created, created))["n"]
    pre = db.row(conn, "SELECT score FROM surveys WHERE user_id=? AND kind='pre'", (uid,))
    feat = {
        "hours_available": round((deadline - created) / 3600, 2),
        "creation_hour": ldt.hour, "creation_dow": ldt.weekday(),
        "hours_to_first_step": round((steps[0]["scheduled_for"] - created) / 3600, 2) if steps else None,
        "planned_minutes": int(planned_minutes), "n_steps": len(steps), "concurrent_open": concurrent,
        "focus_min_prev_week": round(focus / 60, 1),
        "avg_interruptions": round(sess["i"] / sess["n"], 2) if sess["n"] else None,
        "prev_postponements": sum(t["postponements"] for t in past),
        "prev_late_rate": round(sum(labelled) / len(labelled), 3) if labelled else None,
        "baseline_score": pre["score"] if pre else None,
    }
    for t in TASK_TYPES:
        feat[f"type_{t}"] = int(task_type == t)
    return feat


# --------------------------------------------------------------------------------------- task view
def task_view(conn, t):
    t = dict(t)
    t["steps"] = db.rows(conn, "SELECT id,position,title,minutes,scheduled_for,done FROM steps WHERE task_id=? "
                               "ORDER BY position", (t["id"],))
    feat = json.loads(t["features"]) if t.get("features") else None
    t.pop("features", None)
    t.pop("is_demo", None)
    if t["status"] == "open" and feat is not None:
        # live risk: refresh the behavioural part of the features with what we know now
        t["risk"], t["risk_source"] = ml.predict_risk(feat)
        t["risk"] = round(t["risk"], 3)
    t["focus_seconds"] = db.row(conn, "SELECT COALESCE(SUM(focus_seconds),0) s FROM sessions WHERE task_id=?",
                                (t["id"],))["s"]
    return t


# ------------------------------------------------------------------------ notification service (FR5)
def generate_nudges(conn, user):
    """Decide whether any open task deserves a prompt now, based on deadline proximity and risk.

    TMT: make a distant deadline feel nearer. Fogg: supply a prompt and make the first step small.
    """
    uid, ts = user["id"], now()
    created = []
    tasks = db.rows(conn, "SELECT * FROM tasks WHERE user_id=? AND status='open' AND deadline>?", (uid, ts))
    for t in tasks:
        window = max(t["deadline"] - t["created_at"], 1)
        frac = (ts - t["created_at"]) / window
        hours_left = (t["deadline"] - ts) / 3600
        feat = json.loads(t["features"]) if t["features"] else {}
        risk, _ = ml.predict_risk(feat) if feat else (0.4, "none")
        first_step = db.row(conn, "SELECT title,minutes FROM steps WHERE task_id=? AND done=0 ORDER BY position "
                                  "LIMIT 1", (t["id"],))
        started = t["first_start_at"] is not None
        kind, msg = None, None
        small = f"Just {first_step['minutes']} minutes: {first_step['title']}." if first_step else "Start with 10 minutes."

        if not started:
            if hours_left <= 24:
                kind, msg = "urgent", f"\"{t['title']}\" is due in {max(1, round(hours_left))}h and hasn't been started. {small}"
            elif frac >= 0.4 or (risk >= 0.6 and frac >= 0.12):
                pct = int(frac * 100)
                kind = "risk" if risk >= 0.5 else "start"
                msg = (f"{pct}% of the time for \"{t['title']}\" has gone. {small}")
            elif risk >= 0.7 and frac >= 0.05:
                kind, msg = "early", f"\"{t['title']}\" tends to be started late. Make a quick start today. {small}"
        elif hours_left <= 24:
            left = db.row(conn, "SELECT COUNT(*) n FROM steps WHERE task_id=? AND done=0", (t["id"],))["n"]
            if left:
                kind, msg = "due_soon", f"\"{t['title']}\" is due in {max(1, round(hours_left))}h with {left} step(s) left."

        if not kind:
            continue
        gap = 3 * 3600 if (frac > 0.7 or hours_left <= 24) else 8 * 3600
        last = db.row(conn, "SELECT created_at FROM nudges WHERE task_id=? ORDER BY created_at DESC LIMIT 1",
                      (t["id"],))
        if last and ts - last["created_at"] < gap:
            continue
        conn.execute("INSERT INTO nudges(user_id,task_id,kind,message,risk,created_at) VALUES(?,?,?,?,?,?)",
                     (uid, t["id"], kind, msg, round(risk, 3), ts))
        created.append(kind)
        log_event(conn, user, "nudge_created", {"task_id": t["id"], "kind": kind, "risk": round(risk, 3)})
    return created


# ------------------------------------------------------------------------------------ dashboard (FR6)
def dashboard(conn, user):
    uid, tz = user["id"], user["tz_offset_min"]
    ts = now()
    tasks = db.rows(conn, "SELECT * FROM tasks WHERE user_id=?", (uid,))
    sessions = db.rows(conn, "SELECT * FROM sessions WHERE user_id=? AND status!='running'", (uid,))

    # streak: consecutive local days (ending today or yesterday) with >= 5 focused minutes
    days = {}
    for s in sessions:
        d = local_dt(s["started_at"], tz).date()
        days[d] = days.get(d, 0) + s["focus_seconds"]
    active = {d for d, sec in days.items() if sec >= 300}
    today = local_dt(ts, tz).date()
    cursor = today if today in active else today - timedelta(days=1)
    streak = 0
    while cursor in active:
        streak += 1
        cursor -= timedelta(days=1)
    best, run, prev = 0, 0, None
    for d in sorted(active):
        run = run + 1 if prev and (d - prev).days == 1 else 1
        best, prev = max(best, run), d

    week = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        week.append({"date": d.isoformat(), "label": d.strftime("%a"), "minutes": round(days.get(d, 0) / 60)})

    started = [t for t in tasks if t["first_start_at"]]
    delays = [(t["first_start_at"] - t["created_at"]) / 3600 for t in started]
    fracs = [(t["first_start_at"] - t["created_at"]) / max(t["deadline"] - t["created_at"], 1) for t in started]
    finished = [t for t in tasks if t["status"] == "done"]
    on_time = [t for t in finished if t["completed_at"] <= t["deadline"]]
    closed = [t for t in tasks if t["status"] == "done" or t["deadline"] < ts]
    on_time_rate = (len([t for t in closed if t["status"] == "done" and t["completed_at"] <= t["deadline"]]) /
                    len(closed)) if closed else None

    # personal delay patterns: how far into the window the student starts, by weekday of creation and type
    by_dow, by_type = {}, {}
    for t, f in zip(started, fracs):
        d = local_dt(t["created_at"], tz).weekday()
        by_dow.setdefault(d, []).append(f)
        by_type.setdefault(t["task_type"], []).append(f)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pattern_dow = [{"label": names[d], "value": round(100 * sum(v) / len(v)), "n": len(v)}
                   for d, v in sorted(by_dow.items())]
    pattern_type = [{"label": k.capitalize(), "value": round(100 * sum(v) / len(v)), "n": len(v)}
                    for k, v in sorted(by_type.items())]

    late = [ml.label_task(t, ts) for t in tasks]
    late = [x for x in late if x is not None]
    total_focus = sum(s["focus_seconds"] for s in sessions)
    insight = None
    if len(started) >= 3:
        avg = 100 * sum(fracs) / len(fracs)
        insight = (f"On average you start a task {round(avg)}% of the way to its deadline. "
                   + ("Try a 10-minute first step on the day you add a task."
                      if avg > 40 else "That's an early start, keep it up."))

    return {
        "streak": streak, "best_streak": best, "week": week,
        "total_focus_minutes": round(total_focus / 60), "sessions": len(sessions),
        "open_tasks": len([t for t in tasks if t["status"] == "open"]),
        "done_tasks": len(finished), "on_time_done": len(on_time),
        "on_time_rate": None if on_time_rate is None else round(100 * on_time_rate),
        "avg_start_delay_hours": round(sum(delays) / len(delays), 1) if delays else None,
        "late_start_rate": round(100 * sum(late) / len(late)) if late else None,
        "pattern_dow": pattern_dow, "pattern_type": pattern_type, "insight": insight,
        "interruptions_per_session": round(sum(s["interruptions"] for s in sessions) / len(sessions), 1)
        if sessions else None,
    }
