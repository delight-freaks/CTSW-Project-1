/**
 * Backend API Client
 * Connects to FastAPI backend for multimodal counseling data
 */

const API_BASE_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8080';

const BASE_HEADERS: Record<string, string> = {
  'Content-Type': 'application/json',
  'ngrok-skip-browser-warning': 'true',
};

interface CreateSessionResponse {
  session_id: string;
  turn_id: string;
  turn_index: number;
}

interface CreateTurnResponse {
  turn_id: string;
  turn_index: number;
}

interface KeystrokeEvent {
  type: 'keydown' | 'keyup';
  key: string;
  timestamp: number;
  is_delete: boolean;
}

interface KeystrokeRawInput {
  session_id: string;
  turn_id: number;
  events: KeystrokeEvent[];
}

interface DeletedSegment {
  text: string;
  deleted_at: number;
}

interface TextOutput {
  session_id: string;
  turn_id: number;
  final_text: string;
  deleted_segments: DeletedSegment[];
}

interface SilenceEvent {
  session_id: string;
  turn_id: number;
  type: 'silence_event';
  silence_duration_sec: number;
  last_keystroke_at: number;
  context: 'after_llm_response' | 'mid_typing';
  timestamp: number;
}

/**
 * Generate a UUID v4 for user_id
 */
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Create a new counseling session
 */
export async function createSession(): Promise<CreateSessionResponse> {
  const userId = generateUUID();

  const response = await fetch(`${API_BASE_URL}/sessions`, {
    method: 'POST',
    headers: BASE_HEADERS,
    body: JSON.stringify({ user_id: userId }),
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Create a new turn within a session
 */
export async function createTurn(sessionId: string): Promise<CreateTurnResponse> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/turns`, {
    method: 'POST',
    headers: BASE_HEADERS,
  });

  if (!response.ok) {
    throw new Error(`Failed to create turn: ${response.statusText}`);
  }

  return response.json();
}

/**
 * End a counseling session
 */
export async function endSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/end`, {
    method: 'PATCH',
    headers: BASE_HEADERS,
  });

  if (!response.ok) {
    throw new Error(`Failed to end session: ${response.statusText}`);
  }
}

/**
 * Send raw keystroke events to backend
 */
export async function sendKeystrokeData(data: KeystrokeRawInput): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/keystrokes`, {
    method: 'POST',
    headers: BASE_HEADERS,
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`Failed to send keystroke data: ${response.statusText}`);
  }

  console.log('[API] Keystroke data sent:', await response.json());
}

/**
 * Send text output (final text + deleted segments) to backend
 */
export async function sendTextData(data: TextOutput): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/text`, {
    method: 'POST',
    headers: BASE_HEADERS,
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`Failed to send text data: ${response.statusText}`);
  }

  console.log('[API] Text data sent:', await response.json());
}

/**
 * Send silence event to backend
 */
export async function sendSilenceEvent(data: SilenceEvent): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/silence`, {
    method: 'POST',
    headers: BASE_HEADERS,
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`Failed to send silence event: ${response.statusText}`);
  }

  console.log('[API] Silence event sent:', await response.json());
}

export { API_BASE_URL };
