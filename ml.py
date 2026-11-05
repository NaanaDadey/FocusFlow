"""Analytics / ML service (Figure 3.1): predicts whether a task will be started late.

Follows thesis Sections 3.2-3.5:
  * label  : procrastinated = first focus session began after >50% of the creation->deadline window
  * models : baseline rule, logistic regression, random forest, gradient boosting
  * split  : by STUDENT (20% held out), stratified grouped 5-fold CV for tuning
  * SMOTE  : applied inside training folds only, and only if the classes are imbalanced
  * stats  : Wilcoxon signed-rank on CV fold scores, best model vs logistic regression
"""
import json
import os
import time
import warnings

import joblib
import numpy as np
from scipy import stats as sps
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

import db

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAVE_IMBLEARN = True
except Exception:  # pragma: no cover
    HAVE_IMBLEARN = False

MODEL_PATH = os.path.join(os.path.dirname(db.DB_PATH), "risk_model.joblib")
META_PATH = os.path.join(os.path.dirname(db.DB_PATH), "risk_model_meta.json")

# Table 3.3 candidate features. Everything here is known at task-creation time.
FEATURES = [
    "hours_available", "creation_hour", "creation_dow",                       # Timing
    "hours_to_first_step",                                                     # Timing (creation -> first planned start)
    "planned_minutes", "n_steps", "concurrent_open",                           # Workload
    "type_assignment", "type_reading", "type_revision", "type_other",
    "focus_min_prev_week", "avg_interruptions", "prev_postponements", "prev_late_rate",  # Behaviour
    "baseline_score",                                                          # Self-report
]

MIN_TASKS, MIN_STUDENTS = 40, 10
RFE_GRID = [6, 9, 12]


def vectorise(feat):
    return [np.nan if feat.get(k) is None else float(feat[k]) for k in FEATURES]


def label_task(t, now=None):
    """1 = procrastinated, 0 = not, None = not yet determinable."""
    now = now or int(time.time())
    window = max(t["deadline"] - t["created_at"], 1)
    if t["first_start_at"]:
        return int((t["first_start_at"] - t["created_at"]) > 0.5 * window)
    if now > t["deadline"]:
        return 1  # never started before the deadline
    # not started, deadline not reached: already past halfway means it will be a late start
    return 1 if (now - t["created_at"]) > 0.5 * window else None


# ---------------------------------------------------------------------------------------------
# live scoring
# ---------------------------------------------------------------------------------------------
_cache = {"mtime": None, "model": None, "meta": None}


def _load():
    if not os.path.exists(MODEL_PATH):
        return None, None
    m = os.path.getmtime(MODEL_PATH)
    if _cache["mtime"] != m:
        _cache["model"] = joblib.load(MODEL_PATH)
        _cache["meta"] = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
        _cache["mtime"] = m
    return _cache["model"], _cache["meta"]


def baseline_risk(feat):
    """Rule used before a model is trained: short notice and a history of late starts raise risk."""
    risk = 0.35
    ha = feat.get("hours_available")
    if ha is not None:
        risk += 0.25 if ha < 72 else (-0.1 if ha > 240 else 0.05)
    risk += 0.25 * (feat.get("prev_late_rate") or 0)
    risk += 0.03 * min(feat.get("prev_postponements") or 0, 5)
    return float(min(max(risk, 0.05), 0.95))


def predict_risk(feat):
    """Returns (risk 0..1, source string)."""
    model, meta = _load()
    if model is None:
        return baseline_risk(feat), "baseline-rule"
    p = float(model.predict_proba(np.array([vectorise(feat)]))[0, 1])
    return p, meta.get("best_model", "model")


def model_info():
    _, meta = _load()
    return meta or {"best_model": None}


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
def load_dataset(conn, include_demo=True):
    q = ("SELECT t.*, u.code FROM tasks t JOIN users u ON u.id=t.user_id "
         "WHERE t.features IS NOT NULL AND u.consent=1")
    if not include_demo:
        q += " AND t.is_demo=0"
    X, y, groups = [], [], []
    now = int(time.time())
    for t in db.rows(conn, q):
        lab = label_task(t, now)
        if lab is None:
            continue
        X.append(vectorise(json.loads(t["features"])))
        y.append(lab)
        groups.append(t["user_id"])
    return np.array(X, dtype=float), np.array(y), np.array(groups)


def _make_pipeline(estimator, smote):
    """Impute -> standardise -> RFE (logistic estimator, Section 3.4) -> [SMOTE] -> classifier.
    Every step is fitted on training folds only, so nothing leaks from validation/test data."""
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
             ("rfe", RFE(LogisticRegression(max_iter=2000), n_features_to_select=10))]
    if smote and HAVE_IMBLEARN:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=3)))
        steps.append(("clf", estimator))
        return ImbPipeline(steps)
    steps.append(("clf", estimator))
    return SkPipeline(steps)


def _metrics(y, pred, proba):
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
    }
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)) for k, v in out.items()}


def train(include_demo=True, seed=42):
    warnings.filterwarnings("ignore")
    conn = db.connect()
    X, y, groups = load_dataset(conn, include_demo)
    conn.close()
    n_students = len(set(groups.tolist()))
    if len(y) < MIN_TASKS or n_students < MIN_STUDENTS or len(set(y)) < 2:
        raise ValueError(f"Not enough labelled data to train yet: need at least {MIN_TASKS} labelled tasks "
                         f"from {MIN_STUDENTS} consenting students and both classes present "
                         f"(have {len(y)} tasks, {n_students} students).")

    rng = np.random.RandomState(seed)
    students = np.array(sorted(set(groups.tolist())))
    rng.shuffle(students)
    n_test = max(2, int(round(0.2 * len(students))))
    test_students = set(students[:n_test].tolist())
    is_test = np.array([g in test_students for g in groups])
    Xtr, ytr, gtr = X[~is_test], y[~is_test], groups[~is_test]
    Xte, yte = X[is_test], y[is_test]
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        raise ValueError("Train or test split has only one class; collect more data.")

    pos_rate = float(y.mean())
    imbalanced = min(pos_rate, 1 - pos_rate) < 0.35
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    specs = {
        "Logistic regression": (LogisticRegression(max_iter=2000),
                                {"clf__C": [0.01, 0.1, 1, 10, 100], "rfe__n_features_to_select": RFE_GRID}),
        "Random forest": (RandomForestClassifier(random_state=seed, n_jobs=1),
                          {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 5, 8, None],
                           "clf__min_samples_leaf": [1, 3, 5], "rfe__n_features_to_select": RFE_GRID}),
        "Gradient boosting": (GradientBoostingClassifier(random_state=seed),
                              {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.03, 0.1, 0.3],
                               "clf__max_depth": [2, 3, 4], "rfe__n_features_to_select": RFE_GRID}),
    }

    # baseline: predict late whenever hours_available is below the training median
    hi = FEATURES.index("hours_available")
    median_ha = float(np.nanmedian(Xtr[:, hi]))
    base_pred = (Xte[:, hi] < median_ha).astype(int)
    base_score = (median_ha - np.nan_to_num(Xte[:, hi], nan=median_ha))
    results = {"Baseline rule": _metrics(yte, base_pred, base_score)}
    fold_scores, fitted, roc = {}, {}, {}
    fpr, tpr, _ = roc_curve(yte, base_score)
    roc["Baseline rule"] = {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}

    for name, (est, grid) in specs.items():
        pipe = _make_pipeline(est, imbalanced)
        search = RandomizedSearchCV(pipe, grid, n_iter=min(6, int(np.prod([len(v) for v in grid.values()]))),
                                    scoring="f1", cv=cv, random_state=seed, n_jobs=1, error_score=0)
        search.fit(Xtr, ytr, groups=gtr)
        best = search.best_estimator_
        fold_scores[name] = cross_val_score(best, Xtr, ytr, groups=gtr, cv=cv, scoring="f1").tolist()
        proba = best.predict_proba(Xte)[:, 1]
        results[name] = _metrics(yte, (proba >= 0.5).astype(int), proba)
        results[name]["cv_f1_mean"] = round(float(np.mean(fold_scores[name])), 4)
        results[name]["best_params"] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
        fitted[name] = best
        fpr, tpr, _ = roc_curve(yte, proba)
        roc[name] = {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}

    best_name = max(fitted, key=lambda n: (results[n]["f1"] or 0, results[n]["auc"] or 0))

    # Wilcoxon signed-rank on CV fold F1: each tree model vs logistic regression, Holm-corrected (Section 3.5.5)
    raw = {}
    for name in ("Random forest", "Gradient boosting"):
        a, b = fold_scores[name], fold_scores["Logistic regression"]
        try:
            w = sps.wilcoxon(a, b)
            raw[name] = (float(w.statistic), float(w.pvalue))
        except ValueError:
            raw[name] = (None, 1.0)
    order = sorted(raw, key=lambda n: raw[n][1])
    adj, running = {}, 0.0
    for i, n in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * raw[n][1]))
        adj[n] = running
    comparison = {"comparison": f"{best_name} vs Logistic regression", "test": "Wilcoxon signed-rank (Holm-corrected)"}
    if best_name == "Logistic regression":
        comparison.update(test="n/a", statistic=None, p_value=None,
                          note="Logistic regression is the best model; no comparison against itself.")
    else:
        comparison.update(statistic=None if raw[best_name][0] is None else round(raw[best_name][0], 3),
                          p_value=round(adj[best_name], 4), p_value_uncorrected=round(raw[best_name][1], 4),
                          note="Only 5 CV folds: the smallest possible two-sided Wilcoxon p-value is 0.0625, "
                               "so significance at 0.05 cannot be reached with k=5.")
    comparison["effect_size"] = None

    # feature importance of the best model
    clf = fitted[best_name].named_steps["clf"]
    support = fitted[best_name].named_steps["rfe"].support_
    raw_imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    imp = np.zeros(len(FEATURES))
    imp[np.where(support)[0]] = raw_imp
    selected = [f for f, keep in zip(FEATURES, support) if keep]
    importance = sorted(zip(FEATURES, [round(float(v), 4) for v in imp]), key=lambda kv: -kv[1])

    # final model: retrain on ALL students for live scoring (evaluation numbers above are from held-out set)
    live = clone(fitted[best_name])
    live.fit(X, y)
    joblib.dump(live, MODEL_PATH)

    cm = confusion_matrix(yte, (fitted[best_name].predict_proba(Xte)[:, 1] >= 0.5).astype(int)).tolist()
    meta = {
        "trained_at": int(time.time()), "best_model": best_name, "n_tasks": int(len(y)),
        "n_students": int(n_students), "n_train_tasks": int(len(ytr)), "n_test_tasks": int(len(yte)),
        "n_test_students": int(n_test), "positive_rate": round(pos_rate, 4), "imbalanced": bool(imbalanced),
        "smote_used": bool(imbalanced and HAVE_IMBLEARN), "include_demo_data": bool(include_demo),
        "results": results, "comparison": comparison, "selected_features": selected, "importance": importance[:10],
        "confusion_matrix_best": cm, "roc": roc,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f)
    _cache["mtime"] = None
    return meta
