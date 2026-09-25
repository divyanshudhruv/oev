---
title: OEV Demo
emoji: 🎯
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "6.1.0"
app_file: app.py
pinned: false
short_description: Score answer options in one 22ms forward pass
---

OEV is a `184M`-parameter decision engine: state and typed questions in, a calibrated probability distribution over the answer options out, in one forward pass.

Try the preset scenarios, paste your own state as plain text or JSON, and inspect the raw response. The demo loads `student-r2b-oev-tiny.pt` from [divyanshudhruv/oev-typed](https://huggingface.co/divyanshudhruv/oev-typed) on first use.
