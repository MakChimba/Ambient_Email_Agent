# Streaming Demo Capture Checklist

These notes walk through capturing a multi-mode streaming demo for Phase 4/5 sign-off. Follow the steps, drop resulting media into `notebooks/img/` (or a shared drive), and reference the artefacts in the implementation ticket once recorded.

## 1. Environment Prep
- Activate the project virtualenv and export deterministic toggles:
  ```bash
  source .venv/bin/activate
  export EMAIL_ASSISTANT_EVAL_MODE=1
  export HITL_AUTO_ACCEPT=1
  export EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1
  export EMAIL_ASSISTANT_TRACE_DEBUG=1
  export EMAIL_ASSISTANT_TRACE_PROJECT=demo-stream
  ```
- (Optional) set `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1` when you need evaluator-style payloads.
- Ensure `.env` holds `GOOGLE_API_KEY=offline-test-key` (or a real key if you want live output). Leave Gmail creds unset for the offline clip.

## 2. Launch the Stream
- Terminal demo:
  ```bash
  python scripts/run_real_outputs.py --agent-module email_assistant_hitl_memory_gmail --stream --max 1 --respond-only
  ```
  This prints `updates`, `messages`, and `custom` payloads (progress events) to stdout.
- Studio demo (optional alternative):
  ```bash
  langgraph up
  ```
  Open the `email_assistant_hitl_memory_gmail` graph, paste the payload below, set `stream_mode=["updates","messages","custom"]`, and run once to record the UI stream.

## 3. Capture
- **CLI recording**: use `asciinema rec demo.cast --quiet --command "python ... --stream"`, then publish or export GIF via `agg`. Highlight the `custom` channel lines in the summary.
- **Browser recording**: in Chrome, open DevTools → Network → WS, start screen recording (or Performance recording) before hitting “Run.” Capture both the visible stream and the DevTools WebSocket messages showing `updates/messages/custom`.
- Save artefacts to `notebooks/img/` using descriptive names, e.g. `streaming-demo-cli.gif`, `streaming-demo-ui.mp4`. Create thumbnails if needed for README/UPDATES references.

## 4. Test Payload
Use this JSON email (matching the offline dataset) for Studio or direct agent calls:
```json
{
  "id": "thread_demo_001",
  "from": "Alex Gomez <alex@example.com>",
  "to": "me@example.com",
  "subject": "Quick follow-up on the onboarding deck",
  "body": "Hi, could you send me the latest onboarding materials by tomorrow?"
}
```
When invoking manually:
```python
config = {"configurable": {"thread_id": "demo-stream-001"}, "recursion_limit": 80}
agent.invoke({"email_input": EMAIL_JSON}, config=config, durability="sync")
```

## 5. Post-Capture Checklist
- Drop media in repo (or shared drive) and update `notebooks/UPDATES.md` with filenames + short description.
- Link artefacts in `dev_tickets/LangChain-LangGraph-v1-implementation-ticket.md` under Phase 4 “Demo Artifacts.”
- If the clip shows eval-mode output, note that in the ticket so reviewers know the Gmail APIs weren’t exercised.

When ready, Phase 5 handoff should point reviewers to this file plus the recorded assets.
