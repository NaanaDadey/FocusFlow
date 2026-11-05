# Focus Flow

A behaviour-aware web application for reducing academic procrastination among students. It implements the system
described in the thesis *"Focus Flow: Design, Implementation and Evaluation of a Behaviour-Aware Application for
Reducing Academic Procrastination Among Students"* (University of Ghana, Department of Computer Science).

Student app: planner, assignment breakdown, focus sessions, risk-aware nudges, progress dashboard, study questionnaires.
Researcher console: consent-filtered exports, model training and evaluation, and the thesis result tables (4.1 to 4.5).

## Quick start

```bash
pip install -r requirements.txt
python app.py                 # http://localhost:5000
```

| Account | Email | Password |
|---|---|---|
| Demo student (pre-filled history, streak, live nudges) | `demo@focusflow.app` | `demo12345` |
| Researcher | `researcher@focusflow.app` | `researcher123` (override with `FOCUSFLOW_RESEARCHER_PW`) |

Students create their own account on the sign-in page. **Change the researcher password before putting this on a network.**

Run the tests: `python -m unittest discover -s tests -v` (23 tests, about 15 s).

## Architecture (thesis Figure 3.1)

| Thesis component | Implementation |
|---|---|
| Client application (planner, focus session, dashboard) | `static/` single-page app, plain JS, installable as a PWA, phone and desktop layouts |
| Application server (API and business logic) | `app.py` (Flask), `services.py` |
| Database (users, tasks, sessions, events) | SQLite via `db.py` (`data/focusflow.db`, created on first run) |
| Analytics / ML service (risk prediction) | `ml.py`: baseline rule, logistic regression, random forest, gradient boosting |
| Notification service (reminders and nudges) | `services.generate_nudges`, in-app plus browser notifications |

## Requirements traceability (thesis Table 3.4)

| ID | Requirement | Where |
|---|---|---|
| FR1 | Register, sign in, manage profile | `/api/register`, `/api/login`, `/api/profile`; password hashing, HttpOnly session cookie |
| FR2 | Add tasks with deadline and estimated effort | Planner screen, `/api/tasks`, input validation |
| FR3 | Break assignments into scheduled steps | `services.breakdown`: 10-minute first step, 25 to 45 minute blocks, spread across the window, editable |
| FR4 | Timed focus sessions linked to a task | Focus screen: pause, resume, distraction log, breaks, `/api/sessions/*`; sets `first_start_at` |
| FR5 | Reminders and nudges from deadline proximity and predicted risk | `services.generate_nudges` uses `ml.predict_risk`; throttled; browser notifications |
| FR6 | Progress, streaks, personal delay patterns | Progress screen, `services.dashboard` |
| FR7 | Anonymised events recorded with consent | `services.log_event` writes nothing without consent; exports use participant codes only |
| NFR1 | Usability, SUS of at least 68 | SUS questionnaire with standard scoring (`forms.score_sus`) |
| NFR2 | Responsiveness | Every API response carries `X-Response-Time-ms` |
| NFR3 | Security and privacy | Hashed passwords, CSP and security headers, per-user isolation, researcher-only exports, account deletion (right to withdraw) |
| NFR4 | Mobile and desktop browsers | Responsive layout, tested at 390 px and 1180 px |
| NFR5 | Works with intermittent connectivity | Session finish, distraction and step updates queue offline and sync on reconnect |

Feature to driver mapping (Table 3.5) is visible in the UI: breakdown (task too big), focus sessions (distraction),
nudges (distant deadline), streaks (motivation), dashboard (self-awareness).

## Data and ML pipeline (thesis Sections 3.2 to 3.5)

* **Label** (3.2): a task is *procrastinated* if its first focus session starts after more than 50% of the creation to deadline window, or if it is never started before the deadline (`ml.label_task`).
* **Features** (Table 3.3): hours available, hour and weekday of creation, hours to first planned step, planned minutes, number of steps, concurrent open tasks, task type, focus minutes in the previous week, average interruptions, earlier postponements, earlier late-start rate, baseline questionnaire score. All are captured at task creation, so the label cannot leak into them.
* **Split** (3.5.3): by student, 20% of students held out. Stratified grouped 5-fold CV with random search for tuning. Imputation, scaling, RFE feature selection (logistic estimator) and SMOTE are fitted inside training folds only. SMOTE is applied only if the minority class is below 35%.
* **Evaluation** (3.5.4): accuracy, precision, recall, F1, ROC-AUC and confusion matrix on the held-out students, against the median-notice baseline rule.
* **Statistics** (3.5.5): Shapiro-Wilk, then paired t-test or Wilcoxon (one-sided, H1: post is lower than pre), Cohen's d with 95% CI, and Wilcoxon on CV fold F1 for model comparison with Holm correction.
* Until a model is trained, nudges use a transparent rule-based risk score.

## Running the real study

1. Participants register and consent, then complete the **pre-test** (week 0).
2. They use the app normally for six weeks. Events and tasks are logged only for consenting users.
3. At week 6 they complete the **post-test** and the **SUS**.
4. Log in as the researcher, open **Research console**, untick *Include demo data*, click **Train / retrain models**, and read Tables 4.1 to 4.5. Download the CSVs or the Markdown tables.

Training refuses to run until there are at least 40 labelled tasks from 10 consenting students with both classes present.

## Synthetic demo data: read this

The researcher console can generate 40 **synthetic** students (`seed.py`, all rows flagged `is_demo=1`) so you can demonstrate
the dashboards, model training and exports before real participants exist. The generator is an assumption written by the
developer (late starts are more likely on short notice and with higher baseline scores, with a small improvement after
week 3). Any pattern a model "finds" in it is circular. **Never report numbers computed from demo data as study results.**
Exports that include demo data carry a visible warning, and *Include demo data* can be switched off everywhere.

## Things to settle with your supervisor

* **Procrastination scale.** The thesis leaves its name as a placeholder (the literature review cites Tuckman, 1991). The 12 items in `forms.py` are a stand-in written for this project, not a published instrument. Replace `PROCRASTINATION_ITEMS` with the licensed or published items you will actually use; scoring and statistics adapt automatically.
* **Gradient boosting.** The thesis cites XGBoost (Chen and Guestrin, 2016). This build uses scikit-learn's `GradientBoostingClassifier` (same method family, no extra dependency). Swapping in `xgboost.XGBClassifier` is a one-line change in `ml.py`.
* **Five folds and Wilcoxon.** With only 5 CV folds the smallest possible two-sided Wilcoxon p-value is 0.0625, so model comparisons cannot reach p < 0.05. Consider repeated grouped CV (for example 5x5) if you want a testable comparison. The console prints this caveat.
* **NFR2** in the thesis says "under two weeks". This was presumably meant as seconds, and the app responds in milliseconds.
* **Interviews** (thematic analysis) are outside the software. SUS free-text feedback is stored with emails and phone numbers scrubbed.

## Deployment notes

```bash
pip install gunicorn
FOCUSFLOW_HTTPS=1 FOCUSFLOW_RESEARCHER_PW='choose-a-strong-one' gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

Put it behind HTTPS (the thesis requires protection in transit). Back up `data/focusflow.db`. The session secret is generated
into `data/.secret_key` on first run, or set `FOCUSFLOW_SECRET`.

## Layout

```
app.py        Flask API and routes          ml.py        risk model: training, evaluation, serving
services.py   breakdown, features, nudges   stats.py     paired tests and Tables 4.1 to 4.5
db.py         SQLite schema                 forms.py     questionnaires and scoring
seed.py       synthetic demo data           static/      client app (HTML, CSS, JS, PWA manifest)
tests/        23 automated tests            docs/screens screenshots for Figure 4.1
```
