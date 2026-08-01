# Video Truthfulness Scanner

Paste a YouTube / Instagram / TikTok video URL. The app downloads the video,
transcribes speech locally with Whisper, fact-checks the claims via an LLM
(Moonshot Kimi, with OpenAI as a fallback), and screens sampled frames for
signs of AI-generated/manipulated visuals.

## Requirements

- Python 3.10+
- ffmpeg installed and on your PATH

## Setup

cd api-server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# open .env and paste your MOONSHOT_API_KEY (or OPENAI_API_KEY) there

uvicorn main:app --host 0.0.0.0 --port 8080

That's it — one server, no separate frontend build step. Open
http://localhost:8080 in your browser and the app is right there.

## Scan history

Every scan is saved to a local SQLite file (api-server/history.db, created
automatically on first run).

- Paste a URL you've already scanned and the app returns your saved result
  instantly instead of re-scanning. The API supports
  POST /api/scan?force=true to force a fresh re-scan.
- Click History in the header to browse past scans, or search by title or
  by describing something you remember from the video's content/transcript.

## Video Editor

Every scanned video is kept (re-encoded to mp4) in api-server/media/, so
you can edit it right in the app:

- Click Edit Video (appears after a scan) to open the editor
- Drag the start/end sliders to trim the clip
- Optionally add a text caption/overlay
- Click Export Clip to render it and preview/download the result

Disk usage note: stored videos aren't automatically deleted — clean out
api-server/media/ yourself periodically if disk space matters.

## Frontend Builder

Click Frontend Builder in the header to generate a standalone HTML page
from a plain-language description.

- Describe the screen/page you want
- Optionally include context from your last scanned video
- Preview it live, view the raw code, copy it, or download it

## How-to / process check

If the video is a tutorial or how-to, the app also checks whether the
demonstrated method is correct, and gives the correct method if not.

## AI-generated content check

Sampled video frames are screened for signs of AI generation or manipulation.
kimi-k3 has native vision, so a single MOONSHOT_API_KEY powers this. OpenAI
is used as an automatic fallback if both keys are present and Kimi fails.
This is a best-effort signal, not a definitive verdict.

## Notes

- First run downloads the Whisper model (base size) — needs internet.
- Max video length is 20 minutes (MAX_VIDEO_DURATION_SECONDS in main.py).
- Never commit your .env file or paste your API key anywhere public.
