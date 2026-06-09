import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

# --- Config ---
_REQUIRED_ENV_VARS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]

_missing = [var for var in _REQUIRED_ENV_VARS if not os.getenv(var)]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(_missing)}\n"
        f"Please check your .env file. See .env.example for reference."
    )

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     int(os.getenv("DB_PORT", "")),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# --- SQL ---
ENABLE_TIMESCALE = "CREATE EXTENSION IF NOT EXISTS timescaledb;"

CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    baseline_wpm        FLOAT,
    baseline_expression JSONB
);
"""

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(user_id),
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    status      VARCHAR(20) DEFAULT 'active'
                    CHECK (status IN ('active', 'completed', 'abandoned'))
);
"""

CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     UUID REFERENCES sessions(session_id),
    turn_index     INT NOT NULL,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at   TIMESTAMPTZ,
    user_text        TEXT,
    deleted_segments JSONB,
    agent_response   TEXT
);
"""

CREATE_EXPRESSIONS = """
CREATE TABLE IF NOT EXISTS expressions (
    expression_id    UUID DEFAULT gen_random_uuid(),
    turn_id          UUID REFERENCES turns(turn_id),
    session_id       UUID REFERENCES sessions(session_id),
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    frame_index      INT,
    face_detected    BOOLEAN,
    dominant_emotion VARCHAR(20),
    confidence       FLOAT,
    scores           JSONB,
    head_pose        JSONB,
    peak_emotion     VARCHAR(20),
    peak_confidence  FLOAT,
    peak_detected_at TIMESTAMPTZ
);
"""

CREATE_KEYSTROKES = """
CREATE TABLE IF NOT EXISTS keystrokes (
    keystroke_id  UUID DEFAULT gen_random_uuid(),
    turn_id       UUID REFERENCES turns(turn_id),
    session_id    UUID REFERENCES sessions(session_id),
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    key_char      VARCHAR(20),
    event_type    VARCHAR(10),
    is_backspace  BOOLEAN,
    pause_before_ms INT
);
"""

CREATE_KEYSTROKE_CLASSIFIER = """
CREATE TABLE IF NOT EXISTS keystroke_classifier_output (
    classifier_id  UUID DEFAULT gen_random_uuid(),
    turn_id        UUID REFERENCES turns(turn_id),
    session_id     UUID REFERENCES sessions(session_id),
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    emotion        VARCHAR(20),
    confidence     FLOAT,
    avg_iki_ms     FLOAT,
    backspace_rate FLOAT
);
"""

CREATE_SILENCE_EVENTS = """
CREATE TABLE IF NOT EXISTS silence_events (
    silence_id           UUID DEFAULT gen_random_uuid(),
    turn_id              UUID REFERENCES turns(turn_id),
    session_id           UUID REFERENCES sessions(session_id),
    timestamp            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    silence_duration_sec FLOAT,
    last_keystroke_at    TIMESTAMPTZ,
    context              VARCHAR(30)
                             CHECK (context IN ('after_llm_response', 'mid_typing'))
);
"""

CREATE_PROMPT_LOGS = """
CREATE TABLE IF NOT EXISTS prompt_logs (
    log_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id      UUID REFERENCES turns(turn_id),
    session_id   UUID REFERENCES sessions(session_id),
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prompt_sent  TEXT,
    raw_response TEXT,
    model_used   VARCHAR(50),
    latency_ms   INT
);
"""

# Hypertables for time-series tables
HYPERTABLE_EXPRESSIONS = "SELECT create_hypertable('expressions',               'timestamp', if_not_exists => TRUE);"
HYPERTABLE_KEYSTROKES  = "SELECT create_hypertable('keystrokes',                'timestamp', if_not_exists => TRUE);"
HYPERTABLE_SILENCE     = "SELECT create_hypertable('silence_events',            'timestamp', if_not_exists => TRUE);"
HYPERTABLE_CLASSIFIER  = "SELECT create_hypertable('keystroke_classifier_output','timestamp', if_not_exists => TRUE);"


def init_db():
    log.info("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    log.info("Enabling TimescaleDB extension...")
    cur.execute(ENABLE_TIMESCALE)

    log.info("Creating tables...")
    for name, stmt in [
        ("users",                      CREATE_USERS),
        ("sessions",                   CREATE_SESSIONS),
        ("turns",                      CREATE_TURNS),
        ("expressions",                CREATE_EXPRESSIONS),
        ("keystrokes",                 CREATE_KEYSTROKES),
        ("keystroke_classifier_output",CREATE_KEYSTROKE_CLASSIFIER),
        ("silence_events",             CREATE_SILENCE_EVENTS),
        ("prompt_logs",                CREATE_PROMPT_LOGS),
    ]:
        cur.execute(stmt)
        log.info(f"  ✓ {name}")

    log.info("Converting to hypertables...")
    cur.execute(HYPERTABLE_EXPRESSIONS)
    log.info("  ✓ expressions                → hypertable")
    cur.execute(HYPERTABLE_KEYSTROKES)
    log.info("  ✓ keystrokes                 → hypertable")
    cur.execute(HYPERTABLE_CLASSIFIER)
    log.info("  ✓ keystroke_classifier_output → hypertable")
    cur.execute(HYPERTABLE_SILENCE)
    log.info("  ✓ silence_events             → hypertable")

    cur.close()
    conn.close()
    log.info("Database initialized successfully.")


if __name__ == "__main__":
    init_db()