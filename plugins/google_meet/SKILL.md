---
name: google_meet
description: Join a Google Meet call, scrape live captions into a transcript, and do the followup work afterwards. Use when the user asks the agent to sit in on a meeting, take notes, summarize it, or action items from it. v1 is listen-only — the agent cannot speak in the meeting yet.
version: 0.1.0
platforms:
  - linux
  - macos
metadata:
  hermes:
    tags: [meetings, google-meet, transcription]
---

# google_meet

## When to use

The user says any of:

- "join my Meet at <url>"
- "take notes on this meeting"
- "summarize the meeting and send followups"
- "sit in on my standup"

## Prerequisites the user must handle once

```bash
pip install playwright
python -m playwright install chromium
hermes plugins enable google_meet
hermes meet auth          # optional; skips guest-lobby wait
```

Run `hermes meet setup` to see what's missing.

## Flow

1. **Join** — call `meet_join(url=...)` with the full `https://meet.google.com/abc-defg-hij` URL.
   Returns immediately. The bot runs as a subprocess alongside the agent loop.

2. **Announce yourself** — the plugin does NOT do an automatic consent
   announcement. You should say (in the chat of the meeting, via whatever
   surface the user is watching) something like:
   > "A Hermes agent bot is in this call taking notes."

3. **Poll while in-meeting** (optional) — call `meet_status()` every minute
   or two to confirm the bot is still alive, and `meet_transcript(last=20)`
   to see the latest captions. Don't re-read the whole transcript every
   time; use `last` to stay cheap.

4. **Leave** — call `meet_leave()` when the user says the meeting's over,
   or set `duration="30m"` on `meet_join` for auto-leave. This finalizes
   the transcript file.

5. **Follow up** — read the full transcript with `meet_transcript()`,
   summarize it, then do whatever the user asked (draft a recap email,
   post to Slack, file action items as issues, etc.) using your regular
   tools.

## Tool reference

| Tool | Use |
|---|---|
| `meet_join(url, guest_name?, duration?, headed?)` | Start bot in the meeting |
| `meet_status()` | Liveness + transcript progress |
| `meet_transcript(last?)` | Read scraped captions |
| `meet_leave()` | Close bot, finalize transcript |
| `meet_say(text)` | **STUB** — not implemented. Returns an error. |

## Important limits

- Captions are only as good as Google Meet's live captions. English-biased,
  lossy on overlapping speakers, occasionally mis-attributes who said what.
- The bot joins as a guest unless `hermes meet auth` was run. Guest mode
  sits in the lobby until a host admits it — warn the user.
- Only one active meeting per hermes install. A second `meet_join` leaves
  the first.
- Windows is not supported.
- The bot CANNOT speak in the meeting (v1). Don't promise the user that
  it will — tell them you're there to listen and follow up outside the
  call.

## Transcript location

```
$HERMES_HOME/workspace/meetings/<meeting-id>/transcript.txt
```

Status/log files live in the same directory. Safe to read with `read_file`.

## Safety

- Only `https://meet.google.com/` URLs pass the safety gate. Anything else
  is rejected before the subprocess launches.
- No calendar scanning. No auto-dial. The user or agent must provide the
  URL explicitly each time.
