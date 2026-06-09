"""
Counselor Agent — FastAPI Backend Boilerplate
=============================================
Covers:
  1. Session management (create session, create turn, end session)
  2. REST endpoints to receive JSON from frontend modules:
     - POST /keystrokes          (raw keystroke events)
     - POST /keystroke-classify   (classifier output)
     - POST /silence             (silence events)
     - POST /text                (text with deleted segments)
     - POST /vision              (vision JSON — HTTP alternative to file watcher)
  3. Vision enricher (reads raw vision JSON, injects session_id/turn_id from
     a state file, writes enriched JSON for the file watcher)
  4. Vision file watcher (reads enriched JSON and stores to DB)
  5. WebSocket endpoint for pushing LLM responses back to the frontend
  6. DB query functions that pull stored data and package it into the JSON
     format the prompt assembler expects

DBMS: TimescaleDB (PostgreSQL)
Framework: FastAPI + asyncpg
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 경로에 추가하여 pipeline 모듈 임포트 가능하게 함
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.pipeline.prompt_assembler import (
    VisionOutput as PAVisionOutput,
    KeystrokeOutput as PAKeystrokeOutput,
    TextInput as PATextInput,
    SilenceEvent as PASilenceEvent,
    DeletedSegment as PADeletedSegment,
    assemble_prompt,
    SYSTEM_PROMPT,
)
from modules.pipeline.llm_client import call_claude_api
from modules.pipeline.baseline_pipeline import assemble_baseline_prompt

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env file if present (does not override existing environment variables)
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', '')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME', 'counselor_agent')}"
)

# Directory the vision module writes its RAW JSON files to (no session/turn info).
VISION_RAW_DIR = os.getenv("VISION_RAW_DIR", "./vision_raw")

# Directory where ENRICHED vision JSON files go (with session_id/turn_id injected).
# The file watcher reads from here.
VISION_JSON_DIR = os.getenv("VISION_JSON_DIR", "./vision_output")

# How often the file watcher and enricher check for new files (seconds)
VISION_POLL_INTERVAL = float(os.getenv("VISION_POLL_INTERVAL", "1.0"))

# State file that tracks the currently active session and turn.
# The vision enricher reads this to inject session_id/turn_id into raw vision files.
SESSION_STATE_FILE = os.getenv("SESSION_STATE_FILE", "./current_session.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("counselor_backend")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# asyncpg connection pool — initialized on startup
db_pool: Optional[asyncpg.Pool] = None


def get_db() -> asyncpg.Pool:
    """Return the connection pool, raising if it hasn't been initialized yet."""
    assert db_pool is not None, "DB pool not initialized"
    return db_pool


# Active WebSocket connections keyed by session_id.
# When the LLM responds, we push the response to the matching socket.
ws_connections: dict[str, WebSocket] = {}


# ---------------------------------------------------------------------------
# Pydantic models — match interface_spec_EN.md exactly
# ---------------------------------------------------------------------------

# ---- Vision (Module 1) ----

class HeadPose(BaseModel):
    yaw: float
    pitch: float
    roll: float


class VisionOutput(BaseModel):
    timestamp: float
    face_detected: bool
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    emotion_scores: Optional[dict[str, float]] = None
    head_pose: Optional[HeadPose] = None
    peak_emotion: Optional[str] = None
    peak_confidence: Optional[float] = None
    peak_detected_at: Optional[float] = None
    # These two are NOT in the vision module's raw output.
    # They are injected by the vision enricher background task
    # (which reads from the session state file) before the file
    # watcher picks them up.
    session_id: Optional[str] = None
    turn_id: Optional[str] = None


# ---- Keystrokes (Module 2 — raw) ----

class KeystrokeEvent(BaseModel):
    type: str                   # "keydown" or "keyup"
    key: str
    timestamp: float
    is_delete: bool


class KeystrokeRawInput(BaseModel):
    session_id: str
    turn_id: int                # turn_index from frontend
    events: list[KeystrokeEvent]


# ---- Keystroke classifier output (Module 2 — classified) ----

class KeystrokeClassifierOutput(BaseModel):
    session_id: str
    turn_id: int
    emotion: str
    confidence: float
    avg_iki_ms: float
    backspace_rate: float


# ---- Text (Module 3) ----

class DeletedSegment(BaseModel):
    text: str
    deleted_at: float


class TextOutput(BaseModel):
    session_id: str
    turn_id: int
    final_text: str
    deleted_segments: list[DeletedSegment] = []


# ---- Silence (Module 4) ----

class SilenceEvent(BaseModel):
    session_id: str
    turn_id: int
    type: str = "silence_event"
    silence_duration_sec: float
    last_keystroke_at: float
    context: str                # "after_llm_response" or "mid_typing"
    timestamp: float


# ---- Session / Turn management ----

class CreateSessionRequest(BaseModel):
    user_id: str


class CreateTurnRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Session state file helpers
# ---------------------------------------------------------------------------

def write_session_state(session_id: str, turn_id: str, turn_index: int):
    """
    Write the current active session/turn to a JSON file on disk.
    The vision enricher reads this to inject session_id and turn_id
    into the raw vision JSON files.
    """
    state = {
        "session_id": session_id,
        "turn_id": turn_id,
        "turn_index": turn_index,
    }
    state_path = Path(SESSION_STATE_FILE)
    # Write to a temp file then rename for atomicity — prevents the
    # enricher from reading a half-written file.
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp_path, state_path)
    logger.info(f"Session state updated: session={session_id}, turn_index={turn_index}")


def read_session_state() -> Optional[dict]:
    """
    Read the current session state from disk.
    Returns None if no state file exists (no active session).
    """
    state_path = Path(SESSION_STATE_FILE)
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read session state file: {e}")
        return None


def clear_session_state():
    """Remove the session state file when a session ends."""
    state_path = Path(SESSION_STATE_FILE)
    if state_path.exists():
        state_path.unlink()
        logger.info("Session state file cleared")


# ---------------------------------------------------------------------------
# Vision enricher — injects session_id/turn_id into raw vision files
# ---------------------------------------------------------------------------

async def vision_enricher(raw_dir: Path, enriched_dir: Path):
    """
    Polls `raw_dir` for JSON files from the vision module (no session/turn info).
    Reads the current session state, injects session_id and turn_id, then
    writes the enriched file to `enriched_dir` where the file watcher picks it up.
    Moves processed raw files to `raw_dir/processed/`.
    """
    processed_dir = raw_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    while True:
        try:
            state = read_session_state()

            if state is None:
                # No active session — skip this cycle, don't touch files
                await asyncio.sleep(VISION_POLL_INTERVAL)
                continue

            json_files = sorted(raw_dir.glob("*.json"))
            for fpath in json_files:
                try:
                    raw = fpath.read_text(encoding="utf-8")
                    data = json.loads(raw)

                    # Inject session/turn info from state file
                    data["session_id"] = state["session_id"]
                    data["turn_id"] = state["turn_id"]

                    # Write enriched file to the directory the file watcher monitors
                    enriched_path = enriched_dir / fpath.name
                    enriched_path.write_text(
                        json.dumps(data), encoding="utf-8"
                    )

                    # Move raw file to processed
                    fpath.rename(processed_dir / fpath.name)

                except Exception as e:
                    logger.error(f"Error enriching vision file {fpath.name}: {e}")
                    fpath.rename(processed_dir / f"ERROR_{fpath.name}")

        except Exception as e:
            logger.error(f"Vision enricher error: {e}")

        await asyncio.sleep(VISION_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    # Startup
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    logger.info("Database connection pool created")

    # Create directories
    raw_dir = Path(VISION_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    enriched_dir = Path(VISION_JSON_DIR)
    enriched_dir.mkdir(parents=True, exist_ok=True)

    # Start the vision enricher (raw → enriched with session/turn injected)
    enricher_task = asyncio.create_task(vision_enricher(raw_dir, enriched_dir))
    logger.info(f"Vision enricher started — watching {raw_dir.resolve()}")

    # Start the vision file watcher (enriched → DB)
    watcher_task = asyncio.create_task(vision_file_watcher(enriched_dir))
    logger.info(f"Vision file watcher started — watching {enriched_dir.resolve()}")

    # Pre-warm the DDAMFN vision pipeline so the first browser webcam frame does
    # not trigger a slow cold load. The load is heavy (torch + checkpoint +
    # MediaPipe) and must run in a worker thread; doing it lazily inside the
    # request handler blocks the event loop right after boot and surfaces in the
    # browser as "추론 서버 연결 실패" until the model finishes loading.
    async def _prewarm_vision():
        try:
            async with _vision_infer_lock:
                await asyncio.to_thread(_get_vision_pipeline)
            logger.info("VisionPipeline pre-warmed at startup")
        except Exception as e:
            logger.warning(f"Vision pre-warm skipped (lazy load will retry): {e}")

    prewarm_task = asyncio.create_task(_prewarm_vision())

    yield

    # Shutdown
    enricher_task.cancel()
    watcher_task.cancel()
    for task in [enricher_task, watcher_task]:
        try:
            await task
        except asyncio.CancelledError:
            pass
    await db_pool.close()
    logger.info("Shutdown complete")


app = FastAPI(title="Counselor Agent Backend", lifespan=lifespan, debug=True)

_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
_CORS_ORIGINS = [o.strip() for o in _CORS_ORIGINS if o.strip()]
if not _CORS_ORIGINS:
    _CORS_ORIGINS = ["http://localhost:5173", "http://localhost:5174", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.ngrok-free\.(app|dev)",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper — resolve turn_id UUID from (session_id, turn_index)
# ---------------------------------------------------------------------------

async def resolve_turn_id(
    conn: asyncpg.pool.PoolConnectionProxy,
    session_id: str,
    turn_index: int,
) -> str:
    """
    The frontend sends turn_id as an integer (turn_index).
    The DB uses UUID turn_ids.  This looks up the UUID for a given
    session + turn_index, or creates the turn row if it doesn't exist yet.
    """
    row = await conn.fetchrow(
        """
        SELECT turn_id FROM turns
        WHERE session_id = $1 AND turn_index = $2
        """,
        uuid.UUID(session_id),
        turn_index,
    )
    if row:
        return str(row["turn_id"])

    # Turn doesn't exist yet — create it
    new_turn_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO turns (turn_id, session_id, turn_index)
        VALUES ($1, $2, $3)
        """,
        new_turn_id,
        uuid.UUID(session_id),
        turn_index,
    )
    return str(new_turn_id)


# ---------------------------------------------------------------------------
# 1. Session management
# ---------------------------------------------------------------------------

@app.post("/sessions")
async def create_session(req: CreateSessionRequest):
    """Create a new session.  Returns the UUID v4 session_id."""
    session_id = uuid.uuid4()
    turn_id = uuid.uuid4()
    async with get_db().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
            """,
            uuid.UUID(req.user_id),
        )
        await conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, status)
            VALUES ($1, $2, 'active')
            """,
            session_id,
            uuid.UUID(req.user_id),
        )
        # Create the first turn automatically
        await conn.execute(
            """
            INSERT INTO turns (turn_id, session_id, turn_index)
            VALUES ($1, $2, 1)
            """,
            turn_id,
            session_id,
        )

    # Write state file so the vision enricher knows the active session/turn
    write_session_state(str(session_id), str(turn_id), 1)

    return {"session_id": str(session_id), "turn_id": str(turn_id), "turn_index": 1}


@app.post("/sessions/{session_id}/turns")
async def create_turn(session_id: str):
    """Create a new turn within a session.  Auto-increments turn_index."""
    async with get_db().acquire() as conn:
        # Get next turn index
        row = await conn.fetchrow(
            """
            SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_index
            FROM turns WHERE session_id = $1
            """,
            uuid.UUID(session_id),
        )
        next_index = row["next_index"]
        turn_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO turns (turn_id, session_id, turn_index)
            VALUES ($1, $2, $3)
            """,
            turn_id,
            uuid.UUID(session_id),
            next_index,
        )

    # Update state file so the vision enricher uses the new turn
    write_session_state(session_id, str(turn_id), next_index)

    return {"turn_id": str(turn_id), "turn_index": next_index}


@app.patch("/sessions/{session_id}/end")
async def end_session(session_id: str):
    """Mark a session as completed."""
    async with get_db().acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET status = 'completed', ended_at = NOW()
            WHERE session_id = $1
            """,
            uuid.UUID(session_id),
        )

    # Clear state file so the vision enricher stops tagging files
    clear_session_state()

    return {"status": "completed"}


# ---------------------------------------------------------------------------
# 2. REST endpoints — receive module JSON and store to DB
# ---------------------------------------------------------------------------

@app.post("/keystrokes")
async def receive_keystrokes(payload: KeystrokeRawInput):
    """Receive raw keystroke events from the frontend (Module 2 raw)."""
    async with get_db().acquire() as conn:
        turn_id = await resolve_turn_id(conn, payload.session_id, payload.turn_id)

        # Compute pause_before_ms for each event
        prev_ts = None
        for evt in payload.events:
            pause_ms = None
            if prev_ts is not None:
                pause_ms = int((evt.timestamp - prev_ts) * 1000)
            prev_ts = evt.timestamp

            await conn.execute(
                """
                INSERT INTO keystrokes
                    (keystroke_id, turn_id, session_id, timestamp,
                     key_char, event_type, is_backspace, pause_before_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                uuid.uuid4(),
                uuid.UUID(turn_id),
                uuid.UUID(payload.session_id),
                datetime.fromtimestamp(evt.timestamp, tz=timezone.utc),
                evt.key,
                evt.type,
                evt.is_delete,
                pause_ms,
            )
    return {"stored": len(payload.events)}


@app.post("/keystroke-classify")
async def receive_keystroke_classification(payload: KeystrokeClassifierOutput):
    """Receive keystroke classifier output (Module 2 classified)."""
    async with get_db().acquire() as conn:
        turn_id = await resolve_turn_id(conn, payload.session_id, payload.turn_id)
        await conn.execute(
            """
            INSERT INTO keystroke_classifier_output
                (classifier_id, turn_id, session_id, timestamp,
                 emotion, confidence, avg_iki_ms, backspace_rate)
            VALUES ($1, $2, $3, NOW(), $4, $5, $6, $7)
            """,
            uuid.uuid4(),
            uuid.UUID(turn_id),
            uuid.UUID(payload.session_id),
            payload.emotion,
            payload.confidence,
            payload.avg_iki_ms,
            payload.backspace_rate,
        )
    return {"status": "ok"}


@app.post("/silence")
async def receive_silence_event(payload: SilenceEvent):
    """Receive silence event from the frontend silence monitor (Module 4)."""
    async with get_db().acquire() as conn:
        turn_id = await resolve_turn_id(conn, payload.session_id, payload.turn_id)
        await conn.execute(
            """
            INSERT INTO silence_events
                (silence_id, turn_id, session_id, timestamp,
                 silence_duration_sec, last_keystroke_at, context)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            uuid.uuid4(),
            uuid.UUID(turn_id),
            uuid.UUID(payload.session_id),
            datetime.fromtimestamp(payload.timestamp, tz=timezone.utc),
            payload.silence_duration_sec,
            datetime.fromtimestamp(payload.last_keystroke_at, tz=timezone.utc),
            payload.context,
        )
    return {"status": "ok"}


@app.post("/text")
async def receive_text(payload: TextOutput):
    """Receive final text + deleted segments from text capture (Module 3)."""
    async with get_db().acquire() as conn:
        turn_id = await resolve_turn_id(conn, payload.session_id, payload.turn_id)
        deleted_json = json.dumps(
            [seg.model_dump() for seg in payload.deleted_segments]
        )
        await conn.execute(
            """
            UPDATE turns
            SET user_text = $1,
                deleted_segments = $2::jsonb,
                submitted_at = NOW()
            WHERE turn_id = $3
            """,
            payload.final_text,
            deleted_json,
            uuid.UUID(turn_id),
        )
    return {"status": "ok"}


@app.post("/vision")
async def receive_vision(payload: VisionOutput):
    """
    Receive vision output via HTTP (Module 1).
    This is the HTTP alternative to the file watcher.
    Requires session_id and turn_id in the payload.
    """
    if not payload.session_id or not payload.turn_id:
        raise HTTPException(
            status_code=400,
            detail="session_id and turn_id are required when posting vision data via HTTP",
        )
    await _store_vision_output(payload)
    return {"status": "ok"}

@app.get("/sessions/{session_id}/vision/latest")
async def get_latest_vision(session_id: str):
    """
    Return the most recent vision/expression row for a session.
    Used by the frontend LiveAnalysis component to poll real emotion data.
    Returns the latest face-detected expression, or face_detected=False if none found.
    """
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    async with get_db().acquire() as conn:
        latest_expr = await conn.fetchrow(
            """
            SELECT dominant_emotion, confidence, scores, head_pose,
                   peak_emotion, peak_confidence, peak_detected_at, timestamp
            FROM expressions
            WHERE session_id = $1 AND face_detected = TRUE
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            sid,
        )

        if latest_expr:
            return {
                "face_detected": True,
                "timestamp": latest_expr["timestamp"].timestamp(),
                "emotion": latest_expr["dominant_emotion"],
                "confidence": latest_expr["confidence"],
                "emotion_scores": (
                    json.loads(latest_expr["scores"])
                    if latest_expr["scores"]
                    else None
                ),
                "peak_emotion": latest_expr["peak_emotion"],
                "peak_confidence": latest_expr["peak_confidence"],
            }
        else:
            return {
                "face_detected": False,
                "timestamp": None,
                "emotion": None,
                "confidence": None,
                "emotion_scores": None,
                "peak_emotion": None,
                "peak_confidence": None,
            }
        
# ---------------------------------------------------------------------------
# Real-time frame inference — browser webcam frame → DDAMFN → vision_output
# ---------------------------------------------------------------------------

_vision_pipeline = None
_vision_infer_lock = asyncio.Lock()


def _get_vision_pipeline():
    """Lazy-load a single VisionPipeline (DDAMFN++) for on-demand frame inference."""
    global _vision_pipeline
    if _vision_pipeline is None:
        from modules.vision.vision_pipeline import VisionPipeline
        model_path = os.getenv("VISION_MODEL_PATH") or str(
            _PROJECT_ROOT / "modules" / "vision" / "checkpoints" / "ddamfn_rafdb_acc0.9204.pth"
        )
        # device: GPU-accelerated by default for this deployment. The pipeline
        # runs a warmup forward at load and auto-falls-back to CPU if the host's
        # torch build can't run the DDAMFN kernels on this GPU (the "no kernel
        # image is available" error on e.g. RTX 50-series / sm_120 without a
        # cu128 torch), so a mismatched build degrades to CPU instead of 500ing.
        # To force CPU, set VISION_DEVICE=cpu.
        device = os.getenv("VISION_DEVICE") or "cuda"
        _vision_pipeline = VisionPipeline(model_path=model_path, device=device)
        logger.info(
            f"VisionPipeline (frame inference) loaded: {model_path} "
            f"(device={device or 'auto'})"
        )
    return _vision_pipeline


class FrameInput(BaseModel):
    image_b64: str   # base64-encoded JPEG ('data:image/...;base64,' prefix tolerated)


@app.post("/sessions/{session_id}/vision/infer")
async def infer_vision_frame(session_id: str, payload: FrameInput):
    """
    Receive a single webcam frame (base64 JPEG) from the browser, run the DDAMFN
    vision pipeline on it, store the result to the expressions table (so the LLM
    prompt pipeline sees facial emotion), and return the vision_output for
    immediate display in the LiveAnalysis panel.
    """
    import base64
    import numpy as np
    import cv2

    raw = payload.image_b64
    if "," in raw:
        raw = raw.split(",", 1)[1]   # strip 'data:image/jpeg;base64,' prefix
    try:
        arr = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        frame = None
    if frame is None:
        raise HTTPException(status_code=400, detail="invalid image data")

    # MediaPipe FaceMesh / torch inference are not safe under concurrent calls — serialize.
    async with _vision_infer_lock:
        # Load in a worker thread so a cold first call never blocks the event
        # loop (idempotent — returns the cached pipeline once warmed).
        pipeline = await asyncio.to_thread(_get_vision_pipeline)
        output = await asyncio.to_thread(pipeline.process_frame, frame)

    # Persist to DB for the LLM prompt pipeline (best-effort; display works regardless).
    try:
        state = read_session_state()
        if state and state.get("session_id") == session_id and state.get("turn_id"):
            v = VisionOutput(**output, session_id=session_id, turn_id=state["turn_id"])
            await _store_vision_output(v)
    except Exception as e:
        logger.warning(f"vision infer DB store skipped: {e}")

    return output


async def _store_vision_output(v: VisionOutput):
    """Shared logic for storing a single vision output row to the expressions table."""
    async with get_db().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO expressions
                (expression_id, turn_id, session_id, timestamp,
                 face_detected, dominant_emotion, confidence, scores,
                 head_pose, peak_emotion, peak_confidence, peak_detected_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            uuid.uuid4(),
            uuid.UUID(v.turn_id) if v.turn_id else None,
            uuid.UUID(v.session_id) if v.session_id else None,
            datetime.fromtimestamp(v.timestamp, tz=timezone.utc),
            v.face_detected,
            v.emotion,
            v.confidence,
            json.dumps(v.emotion_scores) if v.emotion_scores else None,
            json.dumps(v.head_pose.model_dump()) if v.head_pose else None,
            v.peak_emotion,
            v.peak_confidence,
            (
                datetime.fromtimestamp(v.peak_detected_at, tz=timezone.utc)
                if v.peak_detected_at
                else None
            ),
        )


# ---------------------------------------------------------------------------
# 3. Vision file watcher
# ---------------------------------------------------------------------------

async def vision_file_watcher(watch_dir: Path):
    """
    Polls `watch_dir` for enriched .json files (already containing session_id
    and turn_id, injected by the vision enricher).  Reads each file, stores it
    in the expressions table, then moves it to a `processed/` subdirectory.
    """
    processed_dir = watch_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    while True:
        try:
            json_files = sorted(watch_dir.glob("*.json"))
            for fpath in json_files:
                try:
                    raw = fpath.read_text(encoding="utf-8")
                    data = json.loads(raw)
                    vision = VisionOutput(**data)

                    if not vision.session_id or not vision.turn_id:
                        logger.warning(
                            f"Vision file {fpath.name} missing session_id/turn_id — skipping"
                        )
                        fpath.rename(processed_dir / f"ERROR_{fpath.name}")
                        continue

                    await _store_vision_output(vision)

                    # Move to processed
                    fpath.rename(processed_dir / fpath.name)
                    logger.info(f"Processed vision file: {fpath.name}")

                except Exception as e:
                    logger.error(f"Error processing vision file {fpath.name}: {e}")
                    # Move bad files out of the way so we don't retry forever
                    fpath.rename(processed_dir / f"ERROR_{fpath.name}")

        except Exception as e:
            logger.error(f"Vision file watcher error: {e}")

        await asyncio.sleep(VISION_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# 4. WebSocket — push LLM responses to frontend
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Frontend connects here on session start.
    The backend pushes LLM responses through this socket.
    The frontend can also send messages (e.g. the "send" trigger)
    through here if desired.
    """
    await websocket.accept()
    ws_connections[session_id] = websocket
    logger.info(f"WebSocket connected: session {session_id}")

    try:
        while True:
            # Keep the connection alive.  The frontend can send messages here
            # too — for example, a "send" button press trigger.
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")

            if msg_type == "send_trigger":
                # The user pressed send.  Trigger prompt assembly + LLM call.
                turn_index = message.get("turn_id")  # turn_index from frontend
                try:
                    await handle_send_trigger(session_id, turn_index, websocket)
                except Exception as exc:
                    logger.error(
                        f"handle_send_trigger failed (session={session_id}, turn={turn_index}): {exc}",
                        exc_info=True,
                    )
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "text": f"응답 생성 중 오류가 발생했습니다: {exc}",
                        }))
                    except Exception:
                        pass

            elif msg_type == "silence_trigger":
                # The silence monitor detected silence AND the trigger
                # evaluator decided to intervene.  Build silence prompt.
                turn_index = message.get("turn_id")
                try:
                    await handle_silence_trigger(session_id, turn_index, websocket)
                except Exception as exc:
                    logger.error(
                        f"handle_silence_trigger failed (session={session_id}, turn={turn_index}): {exc}",
                        exc_info=True,
                    )
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "text": f"침묵 응답 생성 중 오류가 발생했습니다: {exc}",
                        }))
                    except Exception:
                        pass

            elif msg_type == "baseline_send_trigger":
                # 베이스라인 모드: 최종 텍스트만으로 LLM 호출
                turn_index = message.get("turn_id")
                try:
                    await handle_baseline_send_trigger(session_id, turn_index, websocket)
                except Exception as exc:
                    logger.error(
                        f"handle_baseline_send_trigger failed (session={session_id}, turn={turn_index}): {exc}",
                        exc_info=True,
                    )
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "text": f"베이스라인 응답 생성 중 오류가 발생했습니다: {exc}",
                        }))
                    except Exception:
                        pass

            # Add more message types as needed

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session {session_id}")
    finally:
        ws_connections.pop(session_id, None)


async def push_to_frontend(session_id: str, message: dict):
    """Send a JSON message to the frontend through the WebSocket."""
    ws = ws_connections.get(session_id)
    if ws:
        await ws.send_text(json.dumps(message))
    else:
        logger.warning(f"No WebSocket connection for session {session_id}")


# ---------------------------------------------------------------------------
# 5. DB dict → prompt assembler dataclass 변환 헬퍼
# ---------------------------------------------------------------------------

def _make_vision(v: Optional[dict]) -> PAVisionOutput:
    """DB에서 꺼낸 vision dict를 PAVisionOutput 데이터클래스로 변환한다."""
    if not v or not v.get("face_detected"):
        return PAVisionOutput(
            timestamp=0.0, face_detected=False,
            emotion=None, confidence=None,
            emotion_scores=None, head_pose=None,
        )
    return PAVisionOutput(
        timestamp=v.get("timestamp") or 0.0,
        face_detected=True,
        emotion=v.get("emotion"),
        confidence=v.get("confidence"),
        emotion_scores=v.get("emotion_scores"),
        head_pose=v.get("head_pose"),
        peak_emotion=v.get("peak_emotion"),
        peak_confidence=v.get("peak_confidence"),
        peak_detected_at=v.get("peak_detected_at"),
    )


def _make_keystroke(k: Optional[dict], session_id: str, turn_index: int) -> PAKeystrokeOutput:
    """DB에서 꺼낸 keystroke classifier dict를 PAKeystrokeOutput 데이터클래스로 변환한다."""
    if not k:
        return PAKeystrokeOutput(
            session_id=session_id, turn_id=turn_index,
            emotion="neutral", confidence=0.0,
            avg_iki_ms=None, backspace_rate=None,
        )
    return PAKeystrokeOutput(
        session_id=k.get("session_id", session_id),
        turn_id=k.get("turn_id", turn_index),
        emotion=k.get("emotion", "neutral"),
        confidence=k.get("confidence", 0.0),
        avg_iki_ms=k.get("avg_iki_ms"),
        backspace_rate=k.get("backspace_rate"),
    )


def _make_text(t: Optional[dict], session_id: str, turn_index: int) -> PATextInput:
    """DB에서 꺼낸 text dict를 PATextInput 데이터클래스로 변환한다."""
    if not t:
        return PATextInput(
            session_id=session_id, turn_id=turn_index,
            final_text="", deleted_segments=[],
        )
    segments = [
        PADeletedSegment(text=s["text"], deleted_at=s["deleted_at"])
        for s in t.get("deleted_segments", [])
    ]
    return PATextInput(
        session_id=t.get("session_id", session_id),
        turn_id=t.get("turn_id", turn_index),
        final_text=t.get("final_text", ""),
        deleted_segments=segments,
    )


# ---------------------------------------------------------------------------
# 6. Trigger handlers — query DB, assemble prompt, call LLM
# ---------------------------------------------------------------------------

async def handle_send_trigger(
    session_id: str, turn_index: int, ws: WebSocket
):
    """
    Called when the user presses send.
    Queries all stored data for this turn, assembles prompt, calls Claude API,
    and pushes the LLM response back to the frontend.
    """
    assembled = await query_turn_data_for_prompt(session_id, turn_index)

    vision = _make_vision(assembled.get("vision"))
    keystroke = _make_keystroke(assembled.get("keystroke"), session_id, turn_index)
    text = _make_text(assembled.get("text"), session_id, turn_index)

    system_prompt, user_prompt = assemble_prompt(vision, keystroke, text)

    t0 = asyncio.get_event_loop().time()
    llm_response = await asyncio.to_thread(call_claude_api, system_prompt, user_prompt)
    latency_ms = int((asyncio.get_event_loop().time() - t0) * 1000)

    turn_id = assembled.get("turn_id", "")
    await store_prompt_log(session_id, turn_id, user_prompt, llm_response, "claude-sonnet-4-6", latency_ms)
    await store_agent_response(turn_id, llm_response)

    await ws.send_text(json.dumps({"type": "llm_response", "text": llm_response}))


async def handle_silence_trigger(
    session_id: str, turn_index: int, ws: WebSocket
):
    """
    Called when the silence trigger fires.
    Queries data, assembles silence prompt, calls Claude API,
    and pushes the LLM response back to the frontend.
    """
    assembled = await query_turn_data_for_prompt(session_id, turn_index)

    async with get_db().acquire() as conn:
        turn_id = await resolve_turn_id(conn, session_id, turn_index)
        silence_row = await conn.fetchrow(
            """
            SELECT silence_duration_sec, context, timestamp
            FROM silence_events
            WHERE turn_id = $1
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            uuid.UUID(turn_id),
        )

    vision = _make_vision(assembled.get("vision"))
    keystroke = _make_keystroke(assembled.get("keystroke"), session_id, turn_index)
    text = _make_text(assembled.get("text"), session_id, turn_index)

    silence = None
    if silence_row:
        silence = PASilenceEvent(
            session_id=session_id,
            turn_id=turn_index,
            type="silence_event",
            silence_duration_sec=silence_row["silence_duration_sec"],
            context=silence_row["context"],
            timestamp=silence_row["timestamp"].timestamp(),
        )

    system_prompt, user_prompt = assemble_prompt(vision, keystroke, text, silence)

    t0 = asyncio.get_event_loop().time()
    llm_response = await asyncio.to_thread(call_claude_api, system_prompt, user_prompt)
    latency_ms = int((asyncio.get_event_loop().time() - t0) * 1000)

    await store_prompt_log(session_id, turn_id, user_prompt, llm_response, "claude-sonnet-4-6", latency_ms)
    await store_agent_response(turn_id, llm_response)

    await ws.send_text(json.dumps({"type": "llm_response", "text": llm_response}))


async def handle_baseline_send_trigger(
    session_id: str, turn_index: int, ws: WebSocket
):
    """
    베이스라인 모드: 최종 전송 텍스트만으로 LLM을 호출한다.
    키스트로크, 비전, 삭제된 텍스트, 침묵 신호는 모두 무시한다.
    멀티모달 파이프라인과의 비교 실험 대조군으로 사용한다.
    """
    async with get_db().acquire() as conn:
        turn_id = await resolve_turn_id(conn, session_id, turn_index)
        turn_row = await conn.fetchrow(
            "SELECT user_text FROM turns WHERE turn_id = $1",
            uuid.UUID(turn_id),
        )

    final_text = (turn_row["user_text"] or "") if turn_row else ""

    system_prompt, user_prompt = assemble_baseline_prompt(final_text)

    t0 = asyncio.get_event_loop().time()
    llm_response = await asyncio.to_thread(call_claude_api, system_prompt, user_prompt)
    latency_ms = int((asyncio.get_event_loop().time() - t0) * 1000)

    await store_prompt_log(session_id, turn_id, user_prompt, llm_response, "baseline-claude-sonnet-4-6", latency_ms)
    await store_agent_response(turn_id, llm_response)

    await ws.send_text(json.dumps({"type": "llm_response", "text": llm_response}))


# ---------------------------------------------------------------------------
# 6. DB query functions — pull stored data for prompt assembler
# ---------------------------------------------------------------------------

async def query_turn_data_for_prompt(
    session_id: str, turn_index: int
) -> dict:
    """
    Query all modality data for a given turn and return it as a dict
    matching the JSON shapes the prompt assembler expects.

    Returns:
    {
        "session_id": "...",
        "turn_index": N,
        "turn_id": "...",
        "text": { ... },          # Module 3 — text output format
        "vision": { ... },         # Module 1 — vision output format
        "keystroke": { ... },      # Module 2 — classifier output format
    }
    """
    async with get_db().acquire() as conn:
        turn_id = await resolve_turn_id(conn, session_id, turn_index)
        sid = uuid.UUID(session_id)
        tid = uuid.UUID(turn_id)

        # ---- Text (Module 3 format) ----
        turn_row = await conn.fetchrow(
            """
            SELECT user_text, deleted_segments
            FROM turns WHERE turn_id = $1
            """,
            tid,
        )
        text_data = None
        if turn_row and turn_row["user_text"] is not None:
            deleted = json.loads(turn_row["deleted_segments"]) if turn_row["deleted_segments"] else []
            text_data = {
                "session_id": session_id,
                "turn_id": turn_index,
                "final_text": turn_row["user_text"],
                "deleted_segments": deleted,
            }

        # ---- Vision (Module 1 format) ----
        # Get the LATEST expression row for this turn (current emotion)
        latest_expr = await conn.fetchrow(
            """
            SELECT dominant_emotion, confidence, scores, head_pose,
                   peak_emotion, peak_confidence, peak_detected_at, timestamp
            FROM expressions
            WHERE turn_id = $1 AND face_detected = TRUE
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            tid,
        )

        vision_data = None
        if latest_expr:
            vision_data = {
                "timestamp": latest_expr["timestamp"].timestamp(),
                "face_detected": True,
                "emotion": latest_expr["dominant_emotion"],
                "confidence": latest_expr["confidence"],
                "emotion_scores": (
                    json.loads(latest_expr["scores"])
                    if latest_expr["scores"]
                    else None
                ),
                "head_pose": (
                    json.loads(latest_expr["head_pose"])
                    if latest_expr["head_pose"]
                    else None
                ),
                "peak_emotion": latest_expr["peak_emotion"],
                "peak_confidence": latest_expr["peak_confidence"],
                "peak_detected_at": (
                    latest_expr["peak_detected_at"].timestamp()
                    if latest_expr["peak_detected_at"]
                    else None
                ),
            }
        else:
            # Check if we had frames but no face detected
            any_frame = await conn.fetchrow(
                "SELECT 1 FROM expressions WHERE turn_id = $1 LIMIT 1", tid
            )
            if any_frame:
                vision_data = {
                    "timestamp": None,
                    "face_detected": False,
                    "emotion": None,
                    "confidence": None,
                    "emotion_scores": None,
                    "head_pose": None,
                    "peak_emotion": None,
                    "peak_confidence": None,
                    "peak_detected_at": None,
                }
            # else: vision_data stays None — no vision data at all for this turn

        # ---- Keystroke classifier (Module 2 classified format) ----
        ks_row = await conn.fetchrow(
            """
            SELECT emotion, confidence, avg_iki_ms, backspace_rate
            FROM keystroke_classifier_output
            WHERE turn_id = $1
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            tid,
        )
        keystroke_data = None
        if ks_row:
            keystroke_data = {
                "session_id": session_id,
                "turn_id": turn_index,
                "emotion": ks_row["emotion"],
                "confidence": ks_row["confidence"],
                "avg_iki_ms": ks_row["avg_iki_ms"],
                "backspace_rate": ks_row["backspace_rate"],
            }

        return {
            "session_id": session_id,
            "turn_index": turn_index,
            "turn_id": turn_id,
            "text": text_data,
            "vision": vision_data,
            "keystroke": keystroke_data,
        }


# ---------------------------------------------------------------------------
# 7. Prompt log storage (call after LLM responds)
# ---------------------------------------------------------------------------

async def store_prompt_log(
    session_id: str,
    turn_id: str,
    prompt_sent: str,
    raw_response: str,
    model_used: str,
    latency_ms: int,
):
    """Store a record in prompt_logs after the LLM responds."""
    async with get_db().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO prompt_logs
                (log_id, turn_id, session_id, timestamp,
                 prompt_sent, raw_response, model_used, latency_ms)
            VALUES ($1, $2, $3, NOW(), $4, $5, $6, $7)
            """,
            uuid.uuid4(),
            uuid.UUID(turn_id),
            uuid.UUID(session_id),
            prompt_sent,
            raw_response,
            model_used,
            latency_ms,
        )


async def store_agent_response(turn_id: str, response_text: str):
    """Update the turns table with the agent's response."""
    async with get_db().acquire() as conn:
        await conn.execute(
            "UPDATE turns SET agent_response = $1 WHERE turn_id = $2",
            response_text,
            uuid.UUID(turn_id),
        )


# ---------------------------------------------------------------------------
# Frontend static file serving (production build)
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    _assets_dir = _FRONTEND_DIST / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for any non-API path (SPA routing)."""
        return FileResponse(str(_FRONTEND_DIST / "index.html"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)