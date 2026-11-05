"""Research statistics (thesis Sections 3.5.4, 3.5.5 and Tables 4.1-4.5)."""
import json
import math
import time

import numpy as np
from scipy import stats as sps

import db
import ml


def _sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def paired_test(pre, post, alternative="less"):
    """Shapiro-Wilk on differences, then paired t-test (or Wilcoxon if normality rejected).
    One-sided: alternative='less' tests post < pre (decrease), 'greater' tests post > pre (increase).
    Cohen's d (paired, dz) with a 95% CI."""
    pre, post = np.array(pre, float), np.array(post, float)
    n = len(pre)
    if n < 3:
        return {"n": n, "note": "Need at least 3 complete pre/post pairs."}
    diff = post - pre
    out = {"n": n, "pre_mean": round(float(pre.mean()), 2), "pre_sd": round(_sd(pre), 2),
           "post_mean": round(float(post.mean()), 2), "post_sd": round(_sd(post), 2),
           "change": round(float(diff.mean()), 2)}
    sw = sps.shapiro(diff) if n >= 3 and np.ptp(diff) > 0 else None
    out["shapiro_p"] = round(float(sw.pvalue), 4) if sw else None
    normal = sw is None or sw.pvalue >= 0.05
    if np.ptp(diff) == 0 and diff[0] == 0:
        out.update(test="none", statistic=None, p_value=1.0, effect_size=0.0, ci=[0.0, 0.0])
        return out
    if normal:
        r = sps.ttest_rel(post, pre, alternative=alternative)
        out.update(test="Paired t-test (one-sided)", statistic=round(float(r.statistic), 3),
                   p_value=round(float(r.pvalue), 4))
    else:
        r = sps.wilcoxon(post, pre, alternative=alternative)
        out.update(test="Wilcoxon signed-rank (one-sided)", statistic=round(float(r.statistic), 3),
                   p_value=round(float(r.pvalue), 4))
    sd = _sd(diff)
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    se_d = math.sqrt(1 / n + d * d / (2 * n))
    out["effect_size"] = round(d, 3)
    out["ci"] = [round(d - 1.96 * se_d, 3), round(d + 1.96 * se_d, 3)]
    out["null_hypothesis"] = "rejected" if out["p_value"] < 0.05 else "not rejected"
    return out


def build_report(include_demo=True):
    conn = db.connect()
    ts = int(time.time())
    dflt = "" if include_demo else " AND is_demo=0"
    users = db.rows(conn, f"SELECT * FROM users WHERE role='student'{dflt}")
    ids = [u["id"] for u in users]
    consenting = [u for u in users if u["consent"]]
    rep = {"generated_at": ts, "includes_demo_data": include_demo}

    # Table 4.1 ------------------------------------------------------------------------------
    tasks = db.rows(conn, f"SELECT * FROM tasks WHERE 1=1{dflt}")
    tasks = [t for t in tasks if t["user_id"] in {u["id"] for u in consenting}]
    labels = [ml.label_task(t, ts) for t in tasks]
    labels = [x for x in labels if x is not None]
    completed = 0
    for u in consenting:
        has_post = db.row(conn, "SELECT 1 FROM surveys WHERE user_id=? AND kind='post'", (u["id"],))
        has_sus = db.row(conn, "SELECT 1 FROM surveys WHERE user_id=? AND kind='sus'", (u["id"],))
        completed += 1 if (has_post and has_sus) else 0

    def dist(key):
        d = {}
        for u in consenting:
            d[u.get(key) or "Not stated"] = d.get(u.get(key) or "Not stated", 0) + 1
        return d
    rep["table_4_1"] = {
        "recruited": len(consenting), "completed": completed,
        "gender": dist("gender"), "level": dist("level"), "programme": dist("programme"),
        "task_records": len(labels),
        "proportion_procrastinated": round(100 * sum(labels) / len(labels), 1) if labels else None,
    }

    # Table 4.2 + 4.5 (model rows) -------------------------------------------------------------
    meta = ml.model_info()
    rep["table_4_2"] = meta.get("results")
    rep["model_meta"] = {k: meta.get(k) for k in ("best_model", "n_tasks", "n_students", "n_test_tasks",
                                                   "positive_rate", "imbalanced", "smote_used", "importance",
                                                   "trained_at", "include_demo_data")}

    # Table 4.3 -------------------------------------------------------------------------------
    pairs = {}
    for u in consenting:
        pre = db.row(conn, "SELECT score FROM surveys WHERE user_id=? AND kind='pre'", (u["id"],))
        post = db.row(conn, "SELECT score FROM surveys WHERE user_id=? AND kind='post'", (u["id"],))
        if pre and post:
            pairs[u["id"]] = (pre["score"], post["score"])
    scale = paired_test([p[0] for p in pairs.values()], [p[1] for p in pairs.values()]) if pairs else \
        {"n": 0, "note": "No complete pre/post pairs yet."}

    # behavioural measures: tasks created in the first vs second half of each student's study period
    early_delay, late_delay, early_ot, late_ot, uids = [], [], [], [], []
    for u in consenting:
        ut = [t for t in tasks if t["user_id"] == u["id"]]
        if not ut:
            continue
        t0 = min(t["created_at"] for t in ut)
        mid = t0 + 3 * 7 * 86400
        a = [t for t in ut if t["created_at"] < mid]
        b = [t for t in ut if t["created_at"] >= mid]

        def d(ts_):
            x = [(t["first_start_at"] - t["created_at"]) / 3600 for t in ts_ if t["first_start_at"]]
            return sum(x) / len(x) if x else None

        def ot(ts_):
            c = [t for t in ts_ if t["status"] == "done" or t["deadline"] < ts]
            return 100 * len([t for t in c if t["status"] == "done" and t["completed_at"] <= t["deadline"]]) / len(c) \
                if c else None
        da, db_, oa, ob = d(a), d(b), ot(a), ot(b)
        if None not in (da, db_, oa, ob):
            early_delay.append(da); late_delay.append(db_); early_ot.append(oa); late_ot.append(ob)
    rep["table_4_3"] = {
        "procrastination_scale": scale,
        "start_delay_hours": paired_test(early_delay, late_delay) if len(early_delay) >= 3 else
        {"n": len(early_delay), "note": "Not enough students with tasks in both halves of the study."},
        "tasks_on_time_pct": paired_test(early_ot, late_ot, alternative="greater") if len(early_ot) >= 3 else
        {"n": len(early_ot), "note": "Not enough students with tasks in both halves of the study."},
        "note": "Behavioural measures compare tasks created in weeks 1-3 with weeks 4+ of each student's use. "
                "Delay and scale scores are tested for a decrease; on-time percentage for an increase.",
    }

    # Table 4.4 SUS ---------------------------------------------------------------------------
    sus = [r["score"] for r in db.rows(conn, "SELECT s.score FROM surveys s JOIN users u ON u.id=s.user_id "
                                              f"WHERE s.kind='sus' AND u.role='student'{dflt.replace('is_demo', 'u.is_demo')}")]
    rep["table_4_4"] = {"n": len(sus)} if not sus else {
        "n": len(sus), "mean": round(float(np.mean(sus)), 1), "sd": round(_sd(sus), 1),
        "median": round(float(np.median(sus)), 1), "pct_at_least_68": round(100 * sum(s >= 68 for s in sus) / len(sus), 1)}

    # Engagement ------------------------------------------------------------------------------
    sess = db.rows(conn, "SELECT user_id, started_at FROM sessions WHERE status!='running'")
    per_user = {}
    for s in sess:
        per_user.setdefault(s["user_id"], []).append(s["started_at"])
    spw, retained = [], 0
    for u in consenting:
        ss = per_user.get(u["id"], [])
        if not ss:
            continue
        span_weeks = max((max(ss) - min(ss)) / (7 * 86400), 1)
        spw.append(len(ss) / min(span_weeks, 6))
        if max(ss) >= min(ss) + 5 * 7 * 86400:
            retained += 1
    rep["engagement"] = {"sessions_per_week": round(float(np.mean(spw)), 1) if spw else None,
                         "retained_week6_pct": round(100 * retained / len(consenting), 1) if consenting else None}

    # Table 4.5 -------------------------------------------------------------------------------
    rep["table_4_5"] = {
        "pre_vs_post_procrastination": scale,
        "best_model_vs_logistic_regression": meta.get("comparison"),
    }
    conn.close()
    return rep


def report_markdown(rep):
    def f(v, nd=2):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    L = ["# Focus Flow results export", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(rep['generated_at']))}", ""]
    if rep["includes_demo_data"]:
        L += ["> **WARNING: this export includes SYNTHETIC DEMO DATA. Do not report these numbers as study results.**", ""]
    t1 = rep["table_4_1"]
    L += ["## Table 4.1: Participants and dataset", "", "| Item | Value |", "|---|---|",
          f"| Participants recruited / completed | {t1['recruited']} / {t1['completed']} |",
          f"| Gender | {t1['gender']} |", f"| Level of study | {t1['level']} |", f"| Programme | {t1['programme']} |",
          f"| Task records after cleaning | {t1['task_records']} |",
          f"| Proportion labelled procrastinated | {f(t1['proportion_procrastinated'], 1)}% |", ""]
    L += ["## Table 4.2: Model performance on the held-out test set", "",
          "| Model | Accuracy | Precision | Recall | F1 | AUC |", "|---|---|---|---|---|---|"]
    for name, m in (rep["table_4_2"] or {}).items():
        L.append(f"| {name} | {f(m['accuracy'])} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['auc'])} |")
    if not rep["table_4_2"]:
        L.append("| (no model trained yet) | | | | | |")
    mm = rep["model_meta"]
    if mm.get("best_model"):
        L += ["", f"Best model: **{mm['best_model']}**. SMOTE used: {mm['smote_used']}. "
                  f"Top features: {', '.join(k for k, _ in (mm['importance'] or [])[:3])}.", ""]
    L += ["## Table 4.3: Before and after use", "", "| Measure | n | Pre mean (SD) | Post mean (SD) | Change |",
          "|---|---|---|---|---|"]
    for label, key in (("Procrastination scale score", "procrastination_scale"),
                       ("Average start delay (hours), weeks 1-3 vs 4+", "start_delay_hours"),
                       ("Tasks completed on time (%), weeks 1-3 vs 4+", "tasks_on_time_pct")):
        d = rep["table_4_3"][key]
        if "pre_mean" in d:
            L.append(f"| {label} | {d['n']} | {d['pre_mean']} ({d['pre_sd']}) | {d['post_mean']} ({d['post_sd']}) | {d['change']} |")
        else:
            L.append(f"| {label} | {d.get('n', 0)} | - | - | - |")
    s = rep["table_4_4"]
    L += ["", "## Table 4.4: Usability (SUS)", "", "| Measure | Value |", "|---|---|"]
    if s.get("mean") is not None:
        L += [f"| Mean SUS score (SD) | {s['mean']} ({s['sd']}) |", f"| Median | {s['median']} |",
              f"| Proportion scoring 68 or above | {s['pct_at_least_68']}% |"]
    else:
        L.append("| (no SUS responses yet) | |")
    e = rep["engagement"]
    L += ["", f"Engagement: {f(e['sessions_per_week'], 1)} focus sessions per week on average; "
              f"{f(e['retained_week6_pct'], 1)}% remained active at week 6.", "",
          "## Table 4.5: Statistical tests", "", "| Comparison | Test | Statistic | p-value | Effect size |",
          "|---|---|---|---|---|"]
    for label, d in (("Pre vs post procrastination score", rep["table_4_5"]["pre_vs_post_procrastination"]),
                     ("Best model vs logistic regression", rep["table_4_5"]["best_model_vs_logistic_regression"] or {})):
        L.append(f"| {label} | {d.get('test', '-')} | {f(d.get('statistic'), 3)} | {f(d.get('p_value'), 4)} | "
                 f"{f(d.get('effect_size'), 3)} |")
    return "\n".join(L) + "\n"
