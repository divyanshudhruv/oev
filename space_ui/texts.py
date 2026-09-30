# Markdown constants shown in the Space UI.
TITLE_MD = """
<div id="header">
<h1>OEV</h1>
<p><b>Typed questions in. Calibrated probabilities out. One forward pass. Nothing is generated.</b></p>
<p class="metrics"><b>184M</b> params · calibrated distributions · <b>0</b> tokens generated</p>
</div>
"""

QS_HELP = """**Question schema** - one JSON object per question name. Two
equivalent forms are accepted: the native one and the Jev/TypeSafe
`criteria` one (so existing laya/Kev clients work as-is).

Native (what the presets use):

```json
{
  "question_name": {
    "type": "choice",          // "choice" | "noul" (yes/no) | "score" (ordinal)
    "options": ["a", "b"],     // choice: the labels; score: level descriptions
    "levels": ["low", "high"], // score only: ordinal level descriptions
    "instructions": "what to decide"
  }
}
```

Jev/TypeSafe `criteria` - descriptions are ignored, the keys become the
options (choice) and the list items become the levels (score):

```json
{
  "question_name": {
    "type": "choice",
    "criteria": {"billing": "invoices, payments", "technical": "bugs, outages"}
  },
  "quality": {
    "type": "score",
    "criteria": ["poor", "mixed", "good"]
  }
}
```
"""

API_DOCS = r"""## Use OEV from code

The Space exposes one public API event, `decide`. Calls use two steps: submit
inputs, then fetch the streamed result by event ID.

```bash
SPACE_URL="https://divyanshudhruv-oev-demo.hf.space"
EVENT_ID=$(curl -sS -X POST "$SPACE_URL/call/decide" \
  -H "Content-Type: application/json" \
  -d '{"data": ["state text", "{\"action\": {\"type\": \"choice\", \"options\": [\"continue\", \"stop\"]}}", 1.0]}' |
  python -c "import json,sys; print(json.load(sys.stdin)['event_id'])")
curl -N "$SPACE_URL/call/decide/$EVENT_ID"
```

The response is an output list. Its first item is a JSON object with one
entry per question. Choice and score questions include `answer`, `confidence`,
and the full `probabilities` distribution. Yes/no questions include `p_yes`

Or with the `gradio_client`:

```python
from gradio_client import Client
client = Client("divyanshudhruv/oev-demo")
result = client.predict(
    state,          # str: plain text or JSON document
    questions,      # str: JSON, {"name": {"type": "choice|noul|score", ...}}
    1.0,            # float: temperature
    api_name="/decide",
)
```

Python (local install):

```python
from oev.infer import OEV
agent = OEV("divyanshudhruv/oev-typed/student-r3-oev-tiny.pt")
result = agent.decide(state, questions)
```

Questions accept both the native schema and the Jev/TypeSafe `criteria`
form (`criteria` map for choice, list for score) - see the schema help
in the Playground tab.
"""
