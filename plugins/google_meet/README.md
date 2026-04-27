# google_meet plugin

Let the hermes agent join a Google Meet call, transcribe it, and do the followup
work afterwards.

## Status

**v1 — transcribe-only.** Joins a Meet URL in a headless Chromium via Playwright,
enables live captions, scrapes them from the DOM, and writes a deduplicated
transcript to `$HERMES_HOME/workspace/meetings/<meeting-id>.txt`. The agent then
has the transcript in context and can take followup actions using the rest of
its tools (send Slack updates, file issues, schedule followups, etc.).

A future v2 will add realtime duplex audio (the agent speaks in the meeting) via
OpenAI Realtime / Gemini Live bridged through BlackHole (macOS) or a PulseAudio
null-sink (Linux). Not included in this PR — the tool surface has a `meet_say`
stub that returns "not yet implemented" so the agent interface is stable.

## Enable

```bash
# 1. install playwright + chromium
pip install playwright
python -m playwright install chromium

# 2. enable the plugin
hermes plugins enable google_meet

# 3. (optional, recommended) save a Google session so the bot doesn't sit in
#    the guest lobby waiting for host approval:
hermes meet auth
```

## Use

From the agent (tool calls):

```
meet_join(url="https://meet.google.com/abc-defg-hij")
meet_status()                       # inCall, captioning, transcript length
meet_transcript(last=20)            # last N caption lines
meet_leave()                        # close browser, finalize transcript file
```

From the CLI:

```bash
hermes meet setup                   # preflight: playwright, chromium, auth
hermes meet join https://meet.google.com/abc-defg-hij [--headed]
hermes meet transcript [--last 20]
hermes meet stop
```

## Explicit-by-design

- Only joins URLs passed in explicitly — no calendar scanning, no auto-dial.
- No automatic consent announcement — the user or agent should tell meeting
  participants that a bot is present.
- Refuses to register on Windows (v1) — audio routing for v2 on Windows is
  painful, and guest-join Chromium on Windows has its own failure modes we
  haven't tested.
- One active meeting per session. A second `meet_join` leaves the first.

## Files

- `plugin.yaml` — manifest.
- `__init__.py` — `register(ctx)` entry point.
- `meet_bot.py` — Playwright bot (spawnable as `python -m plugins.google_meet.meet_bot`).
- `process_manager.py` — subprocess lifecycle + status file I/O.
- `cli.py` — `hermes meet ...` subcommands.
- `tools.py` — agent-facing tool schemas + handlers.
- `SKILL.md` — agent usage reference.

## Future work

- v2: realtime duplex audio via OpenAI Realtime / Gemini Live + BlackHole /
  PulseAudio null-sink. `meet_say(text)` becomes real.
- v3: remote node host — Chrome on a user's Mac, gateway on a Linux box.
  Needs a hermes-wide node-host primitive we don't have yet.
