"""Synthetic demo data.

!! Everything created here is SYNTHETIC and flagged is_demo=1. It exists so the application, the dashboards, the
!! model training pipeline and the researcher exports can be demonstrated before real participants exist.
!! Results computed from it must NEVER be reported as findings of the study. The generating process below
!! (late starts more likely for short-notice tasks and high baseline scores; a small improvement after week 3)
!! is an assumption written by the developer, so any model "discovering" these patterns is circular.
"""
import json
import math
import random
import time

from werkzeug.security import generate_password_hash

import db
import forms
import ml
import services as svc

DAY = 86400
PROGRAMMES = ["Computer Science", "Information Technology", "Business Administration", "Nursing", "Economics",
              "Psychology", "Statistics"]
TITLES = {
    "assignment": ["Essay on {}", "Lab report: {}", "Case study write-up: {}", "Group project section: {}"],
    "reading": ["Read chapters on {}", "Journal article review: {}", "Weekly reading: {}"],
    "revision": ["Revise {} for the quiz", "Past questions: {}", "Exam revision: {}"],
    "other": ["Prepare {} presentation", "Practical write-up: {}"],
}
SUBJECTS = ["data structures", "microeconomics", "research methods", "databases", "statistics", "ethics",
            "networks", "accounting", "operating systems", "marketing", "anatomy", "calculus"]


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def clear_demo():
    conn = db.connect()
    ids = [r["id"] for r in db.rows(conn, "SELECT id FROM users WHERE is_demo=1")]
    for uid in ids:
        for t in ("tasks", "sessions", "events", "surveys", "nudges"):
            conn.execute(f"DELETE FROM {t} WHERE user_id=?", (uid,))
    conn.execute("DELETE FROM users WHERE is_demo=1 AND email NOT IN ('demo@focusflow.app')")
    for p in ("risk_model.joblib", "risk_model_meta.json"):
        import os
        fp = os.path.join(os.path.dirname(db.DB_PATH), p)
        if os.path.exists(fp):
            os.remove(fp)
    conn.commit()
    conn.close()
    ml._cache.update(mtime=None, model=None, meta=None)
    ensure_demo_account()      # keep the demo login usable


def _make_task(conn, user, rng, created, ttype, hours, planned, p_late, now, effect_applied):
    deadline = created + int(hours * 3600)
    window = deadline - created
    title = rng.choice(TITLES[ttype]).format(rng.choice(SUBJECTS))
    steps = svc.breakdown(title, ttype, planned, created, deadline)
    feat = svc.capture_features(conn, user, ttype, planned, created, deadline, steps)
    late = rng.random() < p_late
    never = late and rng.random() < 0.12
    frac = rng.uniform(0.52, 0.95) if late else rng.uniform(0.01, 0.46)
    first = None if never else created + int(frac * window)
    if first is not None and first > now:
        first = None
    cur = conn.execute("INSERT INTO tasks(user_id,title,task_type,created_at,deadline,planned_minutes,features,is_demo,"
                       "postponements) VALUES(?,?,?,?,?,?,?,1,?)",
                       (user["id"], title, ttype, created, deadline, planned, json.dumps(feat),
                        rng.choice([0, 0, 1]) + (1 if late and rng.random() < 0.5 else 0)))
    tid = cur.lastrowid
    svc.save_steps(conn, tid, steps)
    completed_at = None
    if first is not None:
        n_sess = rng.randint(1, 4)
        t = first
        for _ in range(n_sess):
            if t > now:
                break
            planned_m = rng.choice([15, 25, 25, 45])
            focus = int(planned_m * 60 * rng.uniform(0.55, 1.0))
            conn.execute("INSERT INTO sessions(user_id,task_id,started_at,ended_at,planned_minutes,focus_seconds,"
                         "interruptions,status,is_demo) VALUES(?,?,?,?,?,?,?,'finished',1)",
                         (user["id"], tid, t, t + focus, planned_m, focus,
                          min(6, max(0, int(rng.gauss(2.3 if late else 1.2, 1.3)))), ))
            t += int(rng.uniform(0.3, 2.0) * DAY)
        p_on_time = 0.55 if late else 0.9
        if rng.random() < p_on_time:
            completed_at = min(deadline - rng.randint(600, 6 * 3600), now)
            completed_at = max(completed_at, first + 600)
        elif deadline < now:
            completed_at = deadline + rng.randint(1800, 2 * DAY) if rng.random() < 0.6 else None
        if completed_at and completed_at > now:
            completed_at = None
    if completed_at:
        conn.execute("UPDATE tasks SET status='done', completed_at=?, first_start_at=? WHERE id=?",
                     (completed_at, first, tid))
        conn.execute("UPDATE steps SET done=1 WHERE task_id=?", (tid,))
    elif first is not None:
        conn.execute("UPDATE tasks SET first_start_at=? WHERE id=?", (first, tid))
    return tid


def seed_demo(n_students=40, seed=7):
    clear_demo()
    rng = random.Random(seed)
    conn = db.connect()
    now = int(time.time())
    start = now - 42 * DAY
    shared_hash = generate_password_hash("demo12345")
    n_tasks = 0
    for i in range(n_students):
        pre_ans = [min(5, max(1, int(round(rng.gauss(3.1, 0.9))))) for _ in forms.PROCRASTINATION_ITEMS]
        pre = forms.score_procrastination(pre_ans)
        gender = rng.choices(["Female", "Male", "Prefer not to say"], [0.5, 0.45, 0.05])[0]
        cur = conn.execute("INSERT INTO users(email,name,pw_hash,role,code,consent,programme,level,gender,"
                           "tz_offset_min,is_demo,created_at) VALUES(?,?,?,?,?,?,?,?,?,0,1,?)",
                           (f"synthetic{i + 1}@example.test", f"Synthetic {i + 1}", shared_hash, "student",
                            f"S{i + 1:03d}", 1, rng.choice(PROGRAMMES), rng.choice(["Level 200", "Level 300", "Level 400"]),
                            gender, start))
        user = db.row(conn, "SELECT * FROM users WHERE id=?", (cur.lastrowid,))
        t0 = start + int(rng.uniform(0, 2) * DAY)
        conn.execute("INSERT INTO surveys(user_id,kind,answers,score,created_at) VALUES(?,?,?,?,?)",
                     (user["id"], "pre", json.dumps({"answers": pre_ans}), pre, t0))
        times = sorted(rng.uniform(t0, now - 3600) for _ in range(rng.randint(8, 13)))
        for created in times:
            created = int(created)
            ttype = rng.choices(svc.TASK_TYPES, [0.45, 0.2, 0.25, 0.1])[0]
            hours = rng.choice([14, 24, 36, 48, 72, 96, 120, 168, 240, 336]) * rng.uniform(0.8, 1.2)
            planned = int(rng.choice([30, 45, 60, 90, 120, 180, 240, 360]))
            local = svc.local_dt(created, 0)
            z = (-0.9 + 2.6 * (1 - min(hours, 336) / 336) - 1.0 * (hours > 168)
                 + 0.09 * (pre - 36) + 0.004 * (planned - 100) + 0.4 * (local.hour >= 21 or local.hour < 5)
                 + 0.25 * (local.weekday() >= 5) + rng.gauss(0, 0.4))
            effect = created > t0 + 21 * DAY
            if effect:
                z -= 0.5
            _make_task(conn, user, rng, created, ttype, hours, planned, _sigmoid(z), now, effect)
            n_tasks += 1
        post_ans = [min(5, max(1, a + rng.choice([-1, -1, 0, 0, 0, 1]))) for a in pre_ans]
        post = forms.score_procrastination(post_ans)
        sus_ans = [rng.choice([4, 5, 5, 4, 3]) if k % 2 == 0 else rng.choice([1, 2, 2, 1, 3]) for k in range(10)]
        conn.execute("INSERT INTO surveys(user_id,kind,answers,score,created_at) VALUES(?,?,?,?,?)",
                     (user["id"], "post", json.dumps({"answers": post_ans}), post, now - DAY))
        conn.execute("INSERT INTO surveys(user_id,kind,answers,score,created_at) VALUES(?,?,?,?,?)",
                     (user["id"], "sus", json.dumps({"answers": sus_ans, "feedback": ""}), forms.score_sus(sus_ans),
                      now - DAY))
    conn.commit()
    conn.close()
    return {"students": n_students, "tasks": n_tasks}


def ensure_demo_account():
    """Rich history for demo@focusflow.app so the dashboard, streak and nudges are visible on first run."""
    conn = db.connect()
    u = db.row(conn, "SELECT * FROM users WHERE email='demo@focusflow.app'")
    if not u or db.row(conn, "SELECT 1 FROM tasks WHERE user_id=?", (u["id"],)):
        conn.close()
        return
    rng = random.Random(11)
    now = int(time.time())
    conn.execute("INSERT OR IGNORE INTO surveys(user_id,kind,answers,score,created_at) VALUES(?,?,?,?,?)",
                 (u["id"], "pre", json.dumps({"answers": [4, 2, 4, 2, 3, 4, 2, 4, 4, 2, 4, 2]}), 40.0, now - 20 * DAY))
    plan = [  # (days ago created, hours window, type, planned minutes, started?, done?)
        (19, 96, "assignment", 180, True, True), (16, 48, "reading", 60, True, True),
        (13, 120, "revision", 120, True, True), (10, 72, "assignment", 240, False, False),
        (8, 48, "other", 45, True, True), (6, 96, "assignment", 150, True, True), (4, 72, "revision", 90, True, True)]
    for ago, hours, ttype, planned, started, done in plan:
        created = now - ago * DAY
        t = _make_task(conn, u, rng, created, ttype, hours, planned, 0.0 if started else 1.0, now, True)
        if not started:   # a realistic history item: started late (80% into the window) and finished after the deadline
            conn.execute("UPDATE tasks SET first_start_at=created_at+CAST(0.8*(deadline-created_at) AS INTEGER), "
                         "status='done', completed_at=deadline+10800 WHERE id=?", (t,))
            conn.execute("UPDATE steps SET done=1 WHERE task_id=?", (t,))
    # live tasks with future deadlines
    for title, ttype, hours_left, planned, created_ago in (
            ("Database design report", "assignment", 20, 180, 3 * DAY), ("Statistics problem set", "assignment", 60, 120, 1 * DAY),
            ("Revise Operating Systems", "revision", 150, 240, 5 * 3600)):
        created = now - created_ago
        deadline = now + hours_left * 3600
        steps = svc.breakdown(title, ttype, planned, created, deadline)
        feat = svc.capture_features(conn, u, ttype, planned, created, deadline, steps)
        cur = conn.execute("INSERT INTO tasks(user_id,title,task_type,created_at,deadline,planned_minutes,features,is_demo)"
                           " VALUES(?,?,?,?,?,?,?,1)", (u["id"], title, ttype, created, deadline, planned, json.dumps(feat)))
        svc.save_steps(conn, cur.lastrowid, steps)
    # a focus streak that ends today
    for d in range(0, 5):
        for k in range(rng.randint(1, 2)):
            ts = now - d * DAY - k * 5400 - 1800
            focus = rng.randint(1200, 1800)
            conn.execute("INSERT INTO sessions(user_id,task_id,started_at,ended_at,planned_minutes,focus_seconds,"
                         "interruptions,status,is_demo) VALUES(?,?,?,?,?,?,?,'finished',1)",
                         (u["id"], None, ts, ts + focus, 25, focus, rng.randint(0, 2)))
    conn.commit()
    conn.close()
