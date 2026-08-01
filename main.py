"""
Video Truthfulness Scanner — API Server
FastAPI backend that downloads a video, transcribes its speech locally with
faster-whisper, fact-checks the transcript via an LLM (Moonshot Kimi, with
OpenAI GPT-4o-mini as a fallback), and separately screens sampled video
frames for signs of AI generation using a vision-capable model.
"""

import base64
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

import yt_dlp

app = FastAPI(title="Video Truthfulness Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "static"
DB_PATH = Path(__file__).parent / "history.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            video_title TEXT,
            video_duration INTEGER,
            verdict TEXT,
            truthfulness_score INTEGER,
            result_json TEXT NOT NULL,
            scanned_at TEXT NOT NULL,
            has_video INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_url ON scans(url)")
    conn.commit()
    conn.close()


init_db()

MEDIA_DIR = Path(__file__).parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALLOWED_DOMAINS = (
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "tiktok.com",
)

MAX_VIDEO_DURATION_SECONDS = 20 * 60  # 20 minutes
METADATA_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 300

WHISPER_MODEL_SIZE = "base"

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
MOONSHOT_MODEL = "kimi-k3"
MOONSHOT_VISION_MODEL = "kimi-k3"  # kimi-k3 has native vision
OPENAI_TEXT_MODEL = "gpt-4o-mini"
OPENAI_VISION_MODEL = "gpt-4o-mini"  # vision-capable

FRAME_SAMPLE_COUNT = 4

_whisper_model = None  # lazy-loaded singleton


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ScanInput(BaseModel):
    url: str = Field(..., min_length=1)


class ClaimAnalysis(BaseModel):
    claim: str
    verdict: str  # True | Mostly True | Uncertain | Misleading | False
    explanation: str
    correction: Optional[str] = None
    how_to_verify: Optional[str] = None  # constructive next step for False/Misleading claims


class AIContentAnalysis(BaseModel):
    checked: bool  # whether the check actually ran
    is_likely_ai_generated: Optional[bool] = None
    confidence: Optional[int] = None  # 0-100
    reasoning: Optional[str] = None
    indicators: List[str] = []
    note: Optional[str] = None  # e.g. why check was skipped


class ProcessCheck(BaseModel):
    is_howto_or_tutorial: bool = False
    issue_found: bool = False
    issue_description: Optional[str] = None
    correct_method: Optional[str] = None


class ScanResult(BaseModel):
    verdict: str
    truthfulness_score: int
    summary: str
    claims: List[ClaimAnalysis]
    verification_tips: str
    process_check: ProcessCheck
    transcript: str
    video_title: str
    video_duration: int
    ai_content_analysis: AIContentAnalysis
    scan_id: Optional[int] = None
    from_cache: bool = False
    scanned_at: Optional[str] = None
    has_video: bool = False


class HistoryItem(BaseModel):
    id: int
    url: str
    video_title: str
    video_duration: int
    verdict: str
    truthfulness_score: int
    scanned_at: str
    has_video: bool = False


class VideoEditInput(BaseModel):
    start: float = Field(0, ge=0)
    end: float = Field(..., gt=0)
    caption: Optional[str] = None


class UIGenerateInput(BaseModel):
    prompt: str = Field(..., min_length=1)
    context: Optional[str] = None  # e.g. video summary/transcript to ground the design in


class UIGenerateResult(BaseModel):
    html: str


# ---------------------------------------------------------------------------
# Step 1 — URL validation
# ---------------------------------------------------------------------------

def validate_url(url: str) -> None:
    if not re.match(r"^https?://", url.strip(), re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid URL")

    if not any(domain in url.lower() for domain in ALLOWED_DOMAINS):
        raise HTTPException(
            status_code=400,
            detail="Unsupported platform. Use a YouTube, Instagram, or TikTok URL.",
        )


# ---------------------------------------------------------------------------
# Step 2 — Video metadata
# ---------------------------------------------------------------------------

def get_video_info(url: str) -> dict:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": METADATA_TIMEOUT_SECONDS,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(
            status_code=400, detail=f"Could not retrieve video info: {e}"
        )
    except Exception:
        raise HTTPException(
            status_code=500, detail="Timed out while fetching video metadata"
        )

    duration = info.get("duration") or 0
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Video is too long ({duration // 60} min). Max is 20 minutes.",
        )

    if info.get("age_limit", 0) and info["age_limit"] >= 18:
        raise HTTPException(status_code=400, detail="Age-restricted content")

    return info


# ---------------------------------------------------------------------------
# Step 3 — Video download (full video, so we can pull both audio + frames)
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: str) -> str:
    output_template = str(Path(output_dir) / "source.%(ext)s")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "socket_timeout": DOWNLOAD_TIMEOUT_SECONDS,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video download failed: {e}")

    # yt-dlp may produce .mp4, .mkv, .webm depending on source — find whatever landed
    candidates = list(Path(output_dir).glob("source.*"))
    if not candidates:
        raise HTTPException(status_code=500, detail="Video download failed")

    return str(candidates[0])


def extract_audio(video_path: str, output_dir: str) -> str:
    audio_path = str(Path(output_dir) / "audio.mp3")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libmp3lame", "-q:a", "4",
                audio_path,
            ],
            check=True,
            capture_output=True,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e.stderr.decode(errors='ignore')[:300]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")

    if not Path(audio_path).exists():
        raise HTTPException(status_code=500, detail="Audio extraction failed")

    return audio_path


def extract_sample_frames(video_path: str, output_dir: str, duration: int, count: int = FRAME_SAMPLE_COUNT) -> List[str]:
    """Extract `count` evenly-spaced frames as jpg files."""
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(exist_ok=True)

    duration = max(duration, 1)
    frame_paths = []

    for i in range(count):
        timestamp = duration * (i + 1) / (count + 1)  # evenly spaced, avoiding very start/end
        frame_path = frames_dir / f"frame_{i}.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
                    "-frames:v", "1", "-q:v", "2", str(frame_path),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            if frame_path.exists():
                frame_paths.append(str(frame_path))
        except Exception:
            continue  # skip a frame that fails rather than aborting the whole scan

    return frame_paths


# ---------------------------------------------------------------------------
# Step 4 — Transcription (local, free)
# ---------------------------------------------------------------------------

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_audio(path: str) -> str:
    model = get_whisper_model()
    try:
        segments, _ = model.transcribe(path, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


# ---------------------------------------------------------------------------
# Step 5 — LLM fact-checking (text)
# ---------------------------------------------------------------------------

def get_text_llm_client():
    """Returns (OpenAI-compatible client, model name) based on available keys."""
    from openai import OpenAI

    moonshot_key = os.environ.get("MOONSHOT_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if moonshot_key:
        client = OpenAI(api_key=moonshot_key, base_url=MOONSHOT_BASE_URL)
        return client, MOONSHOT_MODEL

    if openai_key:
        client = OpenAI(api_key=openai_key)
        return client, OPENAI_TEXT_MODEL

    raise HTTPException(
        status_code=500,
        detail="No AI API key is configured. Set MOONSHOT_API_KEY or OPENAI_API_KEY.",
    )


FACT_CHECK_SYSTEM_PROMPT = """You are a careful fact-checking assistant. Given a \
video title and its spoken transcript, identify the concrete factual claims made, \
and evaluate each one.

For every claim you mark as "Misleading" or "False", include not just the correct \
information but also a short, practical way the viewer could have checked it \
themselves (e.g. "cross-check the stated figure against the official government/ \
company report" or "reverse-image-search the photo to find its original context"). \
This should be a constructive next step, not just a restatement of the correction.

Separately, determine whether this video is a how-to / tutorial / instructional \
video (e.g. showing how to use an app, device, tool, machine, or perform a \
procedure). If it is, and the demonstrated method or steps are wrong, unsafe, \
outdated, or inefficient, describe the issue and give the correct method clearly \
and practically. If it's not a tutorial, or the method shown is correct, mark \
issue_found as false.

Respond with ONLY valid JSON (no markdown fences, no commentary) matching exactly \
this shape:

{
  "verdict": "Mostly True" | "Partially True" | "Misleading" | "Mostly False" | "False",
  "truthfulness_score": <integer 0-100>,
  "summary": "<1-3 sentence plain-language summary>",
  "verification_tips": "<2-4 sentence general guidance for how the viewer can verify content like this in the future, tailored to the specific topic/type of claims found>",
  "claims": [
    {
      "claim": "<the factual claim as stated>",
      "verdict": "True" | "Mostly True" | "Uncertain" | "Misleading" | "False",
      "explanation": "<why this verdict, in plain language>",
      "correction": "<correct info if verdict is False or Misleading, else null>",
      "how_to_verify": "<practical way to check this specific claim, only if verdict is False or Misleading, else null>"
    }
  ],
  "process_check": {
    "is_howto_or_tutorial": true | false,
    "issue_found": true | false,
    "issue_description": "<what's wrong with the demonstrated method, else null>",
    "correct_method": "<clear step-by-step or plain description of the right way to do it, else null>"
  }
}

Only include claims that are actually factual/checkable — skip opinions, jokes, \
or subjective statements. If there are no checkable claims, return an empty \
claims array with a summary explaining that, and still provide general \
verification_tips relevant to the video's topic."""


def fact_check_transcript(transcript: str, video_title: str) -> dict:
    client, model = get_text_llm_client()

    user_prompt = (
        f"Video title: {video_title}\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Analyze the factual claims in this transcript per the instructions."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned malformed JSON")

    return parsed


# ---------------------------------------------------------------------------
# Step 6 — AI-generated content screening (vision)
# ---------------------------------------------------------------------------

AI_DETECTION_SYSTEM_PROMPT = """You are assisting with a best-effort screen for \
signs that video frames were AI-generated or synthetically manipulated (e.g. \
deepfakes, diffusion-model output, face-swaps). This is NOT a definitive \
determination — visual AI-detection is inherently uncertain, so be calibrated \
and conservative in your confidence.

Look for common indicators: unnatural skin texture or lighting, inconsistent \
shadows/reflections, warped or asymmetric facial features, garbled text/logos \
in the background, unnatural blinking or hair strands, mismatched ears/teeth, \
or an overly smooth "airbrushed" look with no natural imperfections.

Respond with ONLY valid JSON (no markdown fences, no commentary):

{
  "is_likely_ai_generated": true | false,
  "confidence": <integer 0-100, how confident you are in that call>,
  "reasoning": "<2-3 sentence plain-language explanation>",
  "indicators": ["<short indicator>", ...]
}

If frames look like an ordinary, unedited real-world recording, say so plainly \
with is_likely_ai_generated: false and explain why."""


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_frames_for_ai_content(frame_paths: List[str]) -> AIContentAnalysis:
    moonshot_key = os.environ.get("MOONSHOT_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not moonshot_key and not openai_key:
        return AIContentAnalysis(
            checked=False,
            note="AI-image detection requires a vision-capable model. Set "
                 "MOONSHOT_API_KEY (kimi-k3 has native vision) or OPENAI_API_KEY.",
        )

    if not frame_paths:
        return AIContentAnalysis(
            checked=False,
            note="Could not extract any video frames to analyze.",
        )

    from openai import OpenAI

    # Prefer Kimi (kimi-k3 has native vision) since it's the primary provider
    # used elsewhere in this app; fall back to OpenAI if Kimi isn't configured.
    if moonshot_key:
        client = OpenAI(api_key=moonshot_key, base_url=MOONSHOT_BASE_URL)
        model = MOONSHOT_VISION_MODEL
    else:
        client = OpenAI(api_key=openai_key)
        model = OPENAI_VISION_MODEL

    content = [
        {"type": "text", "text": "Analyze these sampled video frames per your instructions."}
    ]
    for frame_path in frame_paths:
        b64 = encode_image_base64(frame_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": AI_DETECTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        parsed = json.loads(raw)
    except Exception as e:
        # If Kimi vision fails for any reason and OpenAI is available, try that as a fallback
        if moonshot_key and openai_key:
            try:
                fallback_client = OpenAI(api_key=openai_key)
                response = fallback_client.chat.completions.create(
                    model=OPENAI_VISION_MODEL,
                    messages=[
                        {"role": "system", "content": AI_DETECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.2,
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
                parsed = json.loads(raw)
            except Exception as e2:
                return AIContentAnalysis(checked=False, note=f"AI-content check failed: {e2}")
        else:
            return AIContentAnalysis(checked=False, note=f"AI-content check failed: {e}")

    return AIContentAnalysis(
        checked=True,
        is_likely_ai_generated=parsed.get("is_likely_ai_generated"),
        confidence=parsed.get("confidence"),
        reasoning=parsed.get("reasoning"),
        indicators=parsed.get("indicators", []),
    )


# ---------------------------------------------------------------------------
# Step 7 — Built-in frontend/UI generator
# ---------------------------------------------------------------------------

UI_GENERATOR_SYSTEM_PROMPT = """You are a skilled frontend developer. Generate a \
complete, single, self-contained HTML page based on the user's request.

Rules:
- Output ONLY raw HTML — no markdown fences, no commentary, no explanation before \
or after.
- The response must start with <!DOCTYPE html> and be a fully valid, complete page.
- Use Tailwind via this CDN script tag in <head>: \
<script src="https://cdn.tailwindcss.com"></script>
- Put any custom CSS in a <style> tag and any JS in a <script> tag, all inside \
this single file — no external file references except the Tailwind CDN.
- Make it visually polished, modern, and responsive. Use realistic placeholder \
content (not "Lorem ipsum") relevant to the request.
- If given "context" describing a video's content, ground the design in that \
context (e.g. if the video showed a broken or confusing app screen, design a \
clear, corrected version of that screen).
- Do not include real people's names, copyrighted brand logos, or copyrighted text."""


def generate_frontend_html(prompt: str, context: Optional[str] = None) -> str:
    client, model = get_text_llm_client()

    user_prompt = prompt
    if context:
        user_prompt = f"Context from a video the user watched:\n{context}\n\nRequest: {prompt}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": UI_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frontend generation failed: {e}")

    raw = re.sub(r"^```(?:html)?\s*|\s*```$", "", raw.strip())

    if "<html" not in raw.lower():
        raise HTTPException(
            status_code=500,
            detail="Model did not return a valid HTML page. Try rephrasing your request.",
        )

    return raw


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


def get_cached_scan(url: str) -> Optional[sqlite3.Row]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM scans WHERE url = ? ORDER BY scanned_at DESC LIMIT 1", (url,)
    ).fetchone()
    conn.close()
    return row


def save_scan_to_history(url: str, result: "ScanResult") -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """INSERT INTO scans (url, video_title, video_duration, verdict,
                               truthfulness_score, result_json, scanned_at, has_video)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            url,
            result.video_title,
            result.video_duration,
            result.verdict,
            result.truthfulness_score,
            result.model_dump_json(),
            datetime.now(timezone.utc).isoformat(),
            1 if result.has_video else 0,
        ),
    )
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id


def persist_video_for_editing(video_path: str, scan_id: int) -> bool:
    """Re-encode the downloaded video to a standard mp4 and store it under
    MEDIA_DIR keyed by scan_id, so it can be streamed back for in-app editing."""
    dest_path = MEDIA_DIR / f"{scan_id}.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-movflags", "+faststart",
                str(dest_path),
            ],
            check=True,
            capture_output=True,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
        return dest_path.exists()
    except Exception:
        return False


@app.get("/api/history", response_model=List[HistoryItem])
def get_history(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, url, video_title, video_duration, verdict, truthfulness_score, scanned_at, has_video "
        "FROM scans ORDER BY scanned_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [HistoryItem(**{**dict(row), "has_video": bool(row["has_video"])}) for row in rows]


@app.get("/api/history/search", response_model=List[HistoryItem])
def search_history(q: str):
    query = f"%{q.strip()}%"
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, url, video_title, video_duration, verdict, truthfulness_score, scanned_at, has_video
           FROM scans
           WHERE video_title LIKE ? OR result_json LIKE ?
           ORDER BY scanned_at DESC LIMIT 50""",
        (query, query),
    ).fetchall()
    conn.close()
    return [HistoryItem(**{**dict(row), "has_video": bool(row["has_video"])}) for row in rows]


@app.get("/api/history/{scan_id}", response_model=ScanResult)
def get_history_item(scan_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found in history")
    result = ScanResult.model_validate_json(row["result_json"])
    result.from_cache = True
    return result


@app.post("/api/scan", response_model=ScanResult)
def scan(payload: ScanInput, force: bool = False):
    validate_url(payload.url)
    clean_url = payload.url.strip()

    if not force:
        cached = get_cached_scan(clean_url)
        if cached:
            result = ScanResult.model_validate_json(cached["result_json"])
            result.from_cache = True
            return result

    info = get_video_info(clean_url)
    video_title = info.get("title", "Unknown title")
    video_duration = int(info.get("duration") or 0)

    tmp_dir = tempfile.mkdtemp(prefix="vvs_")
    try:
        video_path = download_video(clean_url, tmp_dir)
        audio_path = extract_audio(video_path, tmp_dir)
        transcript = transcribe_audio(audio_path)

        if not transcript:
            raise HTTPException(
                status_code=500,
                detail="Transcription produced no text — video may have no speech.",
            )

        analysis = fact_check_transcript(transcript, video_title)

        frame_paths = extract_sample_frames(video_path, tmp_dir, video_duration)
        ai_analysis = analyze_frames_for_ai_content(frame_paths)

        result = ScanResult(
            verdict=analysis.get("verdict", "Uncertain"),
            truthfulness_score=int(analysis.get("truthfulness_score", 50)),
            summary=analysis.get("summary", ""),
            claims=[ClaimAnalysis(**c) for c in analysis.get("claims", [])],
            verification_tips=analysis.get("verification_tips", ""),
            process_check=ProcessCheck(**analysis.get("process_check", {})),
            transcript=transcript,
            video_title=video_title,
            video_duration=video_duration,
            ai_content_analysis=ai_analysis,
            scanned_at=datetime.now(timezone.utc).isoformat(),
        )

        scan_id = save_scan_to_history(clean_url, result)

        # Persist the video itself (separately from the temp dir) so it can
        # be edited afterward. Non-fatal if this fails — scan results still
        # return normally, just without the editing option.
        has_video = persist_video_for_editing(video_path, scan_id)
        if has_video:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE scans SET has_video = 1 WHERE id = ?", (scan_id,))
            conn.commit()
            conn.close()

        result.scan_id = scan_id
        result.has_video = has_video
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/video/{scan_id}")
def get_video(scan_id: int):
    video_path = MEDIA_DIR / f"{scan_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="No stored video for this scan")
    return FileResponse(str(video_path), media_type="video/mp4")


@app.post("/api/video/{scan_id}/edit")
def edit_video(scan_id: int, payload: VideoEditInput):
    source_path = MEDIA_DIR / f"{scan_id}.mp4"
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="No stored video for this scan")

    if payload.end <= payload.start:
        raise HTTPException(status_code=400, detail="End time must be after start time")

    tmp_dir = tempfile.mkdtemp(prefix="vvs_edit_")
    output_path = Path(tmp_dir) / "edited.mp4"

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-ss", str(payload.start),
        "-to", str(payload.end),
        "-i", str(source_path),
    ]

    if payload.caption:
        # Basic centered caption near the bottom of the frame. Uses ffmpeg's
        # built-in default font lookup (no explicit fontfile), which works on
        # most systems with fontconfig installed.
        safe_caption = payload.caption.replace("'", "\u2019").replace(":", "\\:")
        drawtext = (
            f"drawtext=text='{safe_caption}':fontcolor=white:fontsize=28:"
            f"box=1:boxcolor=black@0.5:boxborderw=10:"
            f"x=(w-text_w)/2:y=h-th-40"
        )
        ffmpeg_cmd += ["-vf", drawtext]

    ffmpeg_cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Video editing failed: {e.stderr.decode(errors='ignore')[:300]}",
        )
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Video editing failed: {e}")

    if not output_path.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Video editing failed: no output produced")

    return FileResponse(
        str(output_path),
        media_type="video/mp4",
        filename="edited-clip.mp4",
        background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
    )


@app.post("/api/generate-frontend", response_model=UIGenerateResult)
def generate_frontend(payload: UIGenerateInput):
    html = generate_frontend_html(payload.prompt, payload.context)
    return UIGenerateResult(html=html)
