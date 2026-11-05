"""Questionnaires used in the study: needs survey, procrastination scale (pre/post) and the SUS.

NOTE: the thesis leaves the procrastination scale unnamed. PROCRASTINATION_ITEMS below is a
12-item academic-procrastination scale written for this project (5-point Likert, some items
reverse-scored). If your supervisor wants a published instrument (e.g. Tuckman, 1991), replace the
items in this file; scoring, statistics and the UI pick the change up automatically.
"""

LIKERT_AGREE = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# (text, reverse_scored)
PROCRASTINATION_ITEMS = [
    ("I put off starting assignments until the deadline is close.", False),
    ("I start my study tasks as soon as I plan to.", True),
    ("I waste time on my phone or social media when I should be studying.", False),
    ("I finish assignments well before they are due.", True),
    ("Large assignments feel so big that I avoid starting them.", False),
    ("I often leave revision for the last few days before an exam.", False),
    ("I keep to the study schedule I set for myself.", True),
    ("I find reasons to delay work I find unpleasant.", False),
    ("I rush to complete assignments at the last minute.", False),
    ("I can sit down and concentrate on a study task without being distracted.", True),
    ("I postpone study tasks even when I know it will hurt my results.", False),
    ("I break big tasks into smaller parts so that I can begin them.", True),
]

SUS_ITEMS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]

NEEDS_QUESTIONS = [
    {"id": "hours", "text": "On average, how many hours per week do you study outside lectures?", "type": "choice",
     "options": ["0-5", "6-10", "11-15", "16-20", "20+"]},
    {"id": "causes", "text": "What makes you delay studying? (choose all that apply)", "type": "multi",
     "options": ["Task feels too big or unclear", "Deadline is far away", "Phone / social media",
                 "Low motivation", "Poor planning", "Tiredness", "Other commitments"]},
    {"id": "features", "text": "Which features would help you most? (choose all that apply)", "type": "multi",
     "options": ["Breaking assignments into steps", "Focus timer", "Deadline reminders",
                 "Progress and streaks", "Seeing my delay patterns"]},
    {"id": "tools", "text": "Do you currently use any study or task app?", "type": "choice",
     "options": ["No", "Yes, a to-do app", "Yes, a timer app", "Yes, several"]},
]


def score_procrastination(answers):
    """answers: list of 12 ints 1..5. Returns sum score 12..60 (higher = more procrastination)."""
    if not isinstance(answers, list) or len(answers) != len(PROCRASTINATION_ITEMS):
        raise ValueError("Please answer every question.")
    total = 0
    for a, (_, rev) in zip(answers, PROCRASTINATION_ITEMS):
        a = int(a)
        if not 1 <= a <= 5:
            raise ValueError("Answers must be between 1 and 5.")
        total += (6 - a) if rev else a
    return float(total)


def score_sus(answers):
    """Standard SUS scoring (Brooke, 1996): returns 0..100."""
    if not isinstance(answers, list) or len(answers) != 10:
        raise ValueError("Please answer every question.")
    total = 0
    for i, a in enumerate(answers):
        a = int(a)
        if not 1 <= a <= 5:
            raise ValueError("Answers must be between 1 and 5.")
        total += (a - 1) if i % 2 == 0 else (5 - a)
    return total * 2.5


def public_forms():
    return {
        "procrastination": {"items": [t for t, _ in PROCRASTINATION_ITEMS], "scale": LIKERT_AGREE,
                            "min": 12, "max": 60},
        "sus": {"items": SUS_ITEMS, "scale": LIKERT_AGREE},
        "needs": NEEDS_QUESTIONS,
    }
