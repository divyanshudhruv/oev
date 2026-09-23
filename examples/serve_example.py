"""Worked example: batch decisions against the local HTTP server.

Start the server first:

    pip install -e ".[serve]"
    oev-serve --checkpoint checkpoints_td5/oev-tiny.pt --port 8000

Then run this script. It routes a batch of emails through the Jev-compatible
endpoint, which is what an existing TypeSafe client would look like.
"""

import json
import urllib.request

URL = "http://localhost:8000/v1/systemone"

EMAILS = [
    ("email-101", "Hi, we were billed twice for March. Refund the duplicate or we cancel our plan."),
    ("email-102", "The dashboard shows a 500 error since this morning's deploy."),
    ("email-103", "Can you extend our trial by two weeks? Evaluating with the whole team."),
]

QUESTIONS = {
    "department": {"type": "choice", "options": ["billing", "technical", "sales", "other"]},
    "churn_risk": {"type": "noul"},
    "urgency": {"type": "score", "levels": [1, 2, 3]},
}

for email_id, body in EMAILS:
    payload = json.dumps({
        "model": "oev",
        "state": {"id": email_id, "body": body},
        "questions": QUESTIONS,
    }).encode()

    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        out = json.load(resp)

    print(f"\n{email_id}:")
    for name, answer in out["answers"].items():
        print(f"  {name}: {answer['answer']} (confidence {answer['confidence']:.2f})")
