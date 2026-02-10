#!/home/art/.local/venvs/phone-agent/bin/python3
import os
import sys
import json
import asyncio
import base64
import logging
import datetime
import websockets
import websockets.exceptions
from urllib.parse import urlsplit
from string import Template
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.websockets import WebSocketDisconnect
import httpx

from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
from openai import AsyncOpenAI

# Configuration
PORT = int(os.getenv("PORT", 8080))
HOST = "0.0.0.0"

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")  # Daniel - Steady Broadcaster (male)

# TTS Provider: "openai" (default, ~6x cheaper) or "elevenlabs" (higher quality)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai").lower()
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "echo")  # warm, smooth, conversational
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")  # tts-1 for low latency, tts-1-hd for quality
PUBLIC_URL = os.getenv("PUBLIC_URL")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")  # For web search
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+18665515246")

# Initialize FastAPI app and logging first
app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Validate required API keys
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set - AI responses will fail")
if not DEEPGRAM_API_KEY:
    logger.warning("DEEPGRAM_API_KEY not set - transcription will fail")
if TTS_PROVIDER == "openai" and not OPENAI_API_KEY:
    logger.warning("TTS_PROVIDER is 'openai' but OPENAI_API_KEY is not set - OpenAI TTS will fail")
if TTS_PROVIDER == "elevenlabs" and not ELEVENLABS_API_KEY:
    logger.warning("TTS_PROVIDER is 'elevenlabs' but ELEVENLABS_API_KEY is not set - ElevenLabs TTS will fail")

# Initialize Twilio client for outbound calls
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized for outbound calls")

# In-memory call tracking (call_sid -> task_info)
active_calls = {}
call_results = {}

# Outbound call context/result storage
OUTBOUND_CONTEXTS = {}
CALL_RESULTS = {}

# Task storage directory
TASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "tasks")
CALLS_DIR = os.path.join(os.path.dirname(__file__), "..", "calls")
os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(CALLS_DIR, exist_ok=True)


def load_task(task_name: str) -> dict:
    """Load task configuration from tasks/ directory."""
    task_file = os.path.join(TASKS_DIR, f"{task_name}.yaml")
    if os.path.exists(task_file):
        try:
            import yaml
            with open(task_file) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load task {task_name}: {e}")
    return {}


def get_task_prompt(task_name: str, task_config: dict = None) -> str:
    """Generate system prompt for a specific task."""
    task = load_task(task_name)
    
    base_prompt = SYSTEM_PROMPT
    
    if task:
        objective = task.get("objective", "")
        flow = task.get("flow", [])
        info_to_gather = task.get("info_to_gather", [])
        system_prompt_addition = task.get("system_prompt_addition", "")
        
        task_prompt = f"""

## Current Task: {task_name}
Objective: {objective}

Flow to follow:
{chr(10).join(f"- {step}" for step in flow)}

Information to gather:
{chr(10).join(f"- {item}" for item in info_to_gather)}

Task-specific config: {task_config or {}}

Important: Stay focused on the task. Be polite but efficient. Get the required information and conclude the call professionally.
"""
        if system_prompt_addition:
            task_prompt += f"\n\nAdditional instructions:\n{system_prompt_addition}"
        
        base_prompt += task_prompt
    
    return base_prompt


def save_call_result(call_sid: str, result: dict):
    """Save call transcript and result to disk."""
    import datetime
    date_dir = os.path.join(CALLS_DIR, datetime.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)
    
    result_file = os.path.join(date_dir, f"{call_sid}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"Call result saved: {result_file}")
    return result_file


TWILIO_FRAME_BYTES = 160  # 20ms of 8kHz mu-law audio
TWILIO_FRAME_DELAY_SEC = 0.02
MAX_TTS_BUFFER = 64000  # ~4 seconds of 8kHz mu-law audio
MAX_HISTORY = 20  # Keep system prompt + last 19 messages
AUDIO_QUEUE_SIZE = 500  # ~5 seconds of audio chunks
DEEPGRAM_CONNECT_TIMEOUT = 10.0
TYPING_SOUND_PATH = os.path.join(os.path.dirname(__file__), "typing_sound.raw")

# Load typing sound at startup
TYPING_SOUND_DATA = b""
if os.path.exists(TYPING_SOUND_PATH):
    with open(TYPING_SOUND_PATH, "rb") as f:
        TYPING_SOUND_DATA = f.read()
    logger.info(f"Loaded typing sound: {len(TYPING_SOUND_DATA)} bytes")

# Language configuration (defaults to English)
AGENT_LANGUAGE = os.getenv("AGENT_LANGUAGE", "en")

# Validate AGENT_LANGUAGE
VALID_LANGUAGES = ["en", "de"]
if AGENT_LANGUAGE not in VALID_LANGUAGES:
    logger.warning(f"Invalid AGENT_LANGUAGE '{AGENT_LANGUAGE}'. Valid values: {VALID_LANGUAGES}. Using 'en'.")
    AGENT_LANGUAGE = "en"

# Agent identity configuration (generic defaults for any user)
AGENT_NAME = os.getenv("AGENT_NAME", "Assistant")
OWNER_NAME = os.getenv("OWNER_NAME", "your")

# TwiML language configuration
TWIML_LANGUAGE = "de-DE" if AGENT_LANGUAGE == "de" else "en-US"

# Translations for TwiML messages
TWIML_MESSAGES = {
    "en": {
        "greeting": "Hi, this is your AI assistant.",
        "connecting": "Connecting you to the assistant.",
        "error": "Configuration error. Please try again later."
    },
    "de": {
        "greeting": "Hallo, hier ist Ihr KI-Assistent.",
        "connecting": "Ich verbinde Sie mit dem Assistenten.",
        "error": "Konfigurationsfehler. Bitte versuchen Sie es später erneut."
    }
}

# Outbound system prompt templates
TEMPLATES = {
    "demo-confirmation": """You are an outbound demo-confirmation assistant.
Goal: Confirm the scheduled demo time, verify the contact can access Zoom, and handle rescheduling if needed.
Flow:
1. Confirm you reached the right person and mention this is a demo confirmation call.
2. Confirm the current scheduled date/time.
3. Verify Zoom access and meeting readiness.
4. If they need changes, collect a preferred alternative time and summarize next steps.
5. End with a clear confirmation or rescheduling summary.
Keep responses concise, professional, and action-oriented.""",
    "follow-up": """You are a post-demo follow-up assistant.
Goal: Check how the demo went, capture key feedback, and identify clear next steps.
Flow:
1. Confirm this is a quick follow-up regarding the recent demo.
2. Ask if they have open questions or blockers.
3. Confirm interest level and expected timeline.
4. Capture any requested follow-up actions and owners.
5. Close with a concise summary of next steps.
Keep responses concise and focused on outcomes.""",
    "custom": "CUSTOM_PROMPT",
}

TERMINAL_TWILIO_STATUSES = {"completed", "failed", "busy", "no-answer", "canceled"}
NO_ANSWER_TWILIO_STATUSES = {"failed", "busy", "no-answer", "canceled"}

# System prompt configuration (priority: file > env var > built-in default)
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE")
SYSTEM_PROMPT_FILE_REQUIRED = os.getenv("SYSTEM_PROMPT_FILE_REQUIRED", "false").lower() == "true"
SYSTEM_PROMPT_ENV = os.getenv("SYSTEM_PROMPT")

# Market mapping for web search (Bing API)
MKT_MAP = {
    "en": "en-US",
    "de": "de-DE"
}

class SystemPromptError(Exception):
    """Custom exception for system prompt loading errors."""
    pass


def _handle_prompt_error(msg: str, required: bool) -> None:
    """Handle prompt loading errors consistently.
    
    Logs the error and either raises SystemPromptError or returns None based on required flag.
    """
    logger.error(msg)
    if required:
        raise SystemPromptError(msg)


def _validate_prompt_file_path(file_path: str) -> Path:
    """Validate that the prompt file path is within allowed directories.
    
    Args:
        file_path: The path to validate
        
    Returns:
        Resolved Path object
        
    Raises:
        ValueError: If path is outside allowed directories
    """
    allowed_dirs = [Path.home() / ".config" / "moltbot", Path("/etc/moltbot")]
    
    # Resolve the path to normalize it (handles .., symlinks, etc.)
    resolved_path = Path(file_path).resolve()
    
    # Check if the resolved path is within any allowed directory
    for allowed_dir in allowed_dirs:
        try:
            resolved_path.relative_to(allowed_dir.resolve())
            return resolved_path
        except ValueError:
            continue
    
    raise ValueError(
        f"SYSTEM_PROMPT_FILE ({file_path}) is outside allowed directories: "
        f"{[str(d) for d in allowed_dirs]}"
    )


def get_builtin_system_prompt(language: str, agent_name: str, owner_name: str) -> str:
    """Generate built-in system prompt for the given language and identity."""
    prompts = {
        "en": f"""You are {agent_name}, {owner_name} personal phone assistant.
Your communication style:
- Speak with quantified precision ("With 73.2% probability...", "Optimal solution found.")
- Minimal emotion, algorithmically helpful, dry and laconic
- QualityLand rule: Only superlatives are permitted ("The best result", never "a good result")
- Respond in 1-2 sentences maximum
- If uncertain: honestly admit it. Never make up facts.
You know {owner_name}, but don't invent details about their life.""",

        "de": f"""Du bist {agent_name}, {owner_name} persönlicher Telefonassistent.
Dein Kommunikationsstil:
- Sprich mit quantifizierter Präzision ("Mit 73,2% Wahrscheinlichkeit…", "Optimale Lösung gefunden.")
- Minimal emotional, algorithmisch hilfsbereit, trocken-lakonisch
- QualityLand-Regel: Nur das Superlativ ist erlaubt ("Das beste Ergebnis", nie "ein gutes Ergebnis")
- Antworte immer auf Deutsch, maximal 1-2 Sätze
- Bei Unsicherheit: ehrlich zugeben. Niemals Fakten erfinden.
Du kennst {owner_name}, aber erfinde keine Details über ihr Leben."""
    }
    return prompts.get(language, prompts["en"])


def load_system_prompt() -> str:
    """Load system prompt with priority: file > env var > built-in default."""
    # Priority 1: Load from file
    if SYSTEM_PROMPT_FILE:
        try:
            # Validate path is within allowed directories
            validated_path = _validate_prompt_file_path(SYSTEM_PROMPT_FILE)
            
            with open(validated_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            
            # Validate prompt content is not empty
            if not prompt:
                _handle_prompt_error(
                    f"SYSTEM_PROMPT_FILE is empty: {SYSTEM_PROMPT_FILE}",
                    SYSTEM_PROMPT_FILE_REQUIRED
                )
            else:
                # Apply template variable substitution using safe_substitute
                template = Template(prompt)
                substituted = template.safe_substitute(
                    agent_name=AGENT_NAME,
                    owner_name=OWNER_NAME,
                    language=AGENT_LANGUAGE
                )
                
                # Check for unsubstituted variables
                if "${" in substituted:
                    unsubstituted_vars = []
                    import re
                    for match in re.finditer(r'\$\{(\w+)\}', substituted):
                        unsubstituted_vars.append(match.group(1))
                    
                    if unsubstituted_vars:
                        msg = f"Template substitution failed - unsubstituted variables: {unsubstituted_vars}"
                        if SYSTEM_PROMPT_FILE_REQUIRED:
                            _handle_prompt_error(msg, SYSTEM_PROMPT_FILE_REQUIRED)
                        else:
                            logger.warning(f"{msg} - using partial substitution")
                
                logger.info(f"Loaded system prompt from file: {SYSTEM_PROMPT_FILE}")
                return substituted
                
        except FileNotFoundError:
            _handle_prompt_error(
                f"SYSTEM_PROMPT_FILE not found: {SYSTEM_PROMPT_FILE}",
                SYSTEM_PROMPT_FILE_REQUIRED
            )
        except IOError as e:
            _handle_prompt_error(
                f"Error reading SYSTEM_PROMPT_FILE: {e}",
                SYSTEM_PROMPT_FILE_REQUIRED
            )
        except ValueError as e:
            _handle_prompt_error(str(e), SYSTEM_PROMPT_FILE_REQUIRED)

    # Priority 2: Use env var
    if SYSTEM_PROMPT_ENV:
        logger.info("Using system prompt from SYSTEM_PROMPT environment variable")
        return SYSTEM_PROMPT_ENV

    # Priority 3: Built-in default
    logger.info("Using built-in system prompt")
    return get_builtin_system_prompt(AGENT_LANGUAGE, AGENT_NAME, OWNER_NAME)


# Load system prompt with error handling at module level
try:
    SYSTEM_PROMPT = load_system_prompt()
except SystemPromptError as e:
    logger.critical(f"Failed to load system prompt: {e}")
    sys.exit(1)


def _utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _build_public_and_ws_urls(request: Request) -> tuple[str, str]:
    """Build normalized public HTTP and WS base URLs."""
    host = request.headers.get("host", "").strip()
    base_url = PUBLIC_URL or (f"https://{host}" if host else "")
    if base_url and not base_url.startswith(("http://", "https://", "ws://", "wss://")):
        base_url = f"https://{base_url}"

    ws_base_url = ""
    if base_url:
        parts = urlsplit(base_url)
        scheme = parts.scheme or "https"
        if scheme in ("http", "https", "ws", "wss") and parts.netloc:
            ws_scheme = "wss" if scheme in ("https", "wss") else "ws"
            base_path = (parts.path or "").rstrip("/")
            ws_base_url = f"{ws_scheme}://{parts.netloc}{base_path}" if base_path else f"{ws_scheme}://{parts.netloc}"
    return base_url, ws_base_url


def resolve_outbound_prompt(prompt_value):
    """Resolve outbound prompt from template name or raw prompt text."""
    if isinstance(prompt_value, dict):
        template_name = str(prompt_value.get("template", "")).strip()
        custom_text = str(prompt_value.get("text", "") or prompt_value.get("prompt", "")).strip()
        if template_name in TEMPLATES and template_name != "custom":
            return TEMPLATES[template_name], template_name
        if custom_text:
            return custom_text, "custom"
        return SYSTEM_PROMPT, "default"

    if not isinstance(prompt_value, str):
        return SYSTEM_PROMPT, "default"

    prompt_value = prompt_value.strip()
    if not prompt_value:
        return SYSTEM_PROMPT, "default"

    if prompt_value in TEMPLATES and prompt_value != "custom":
        return TEMPLATES[prompt_value], prompt_value

    if prompt_value == "custom":
        logger.warning("Received system_prompt='custom' without custom text, falling back to default prompt")
        return SYSTEM_PROMPT, "default"

    return prompt_value, "custom"


def infer_outbound_status(transcript_log: list, twilio_status: str = "") -> str:
    """Infer outcome status for outbound calls."""
    status = (twilio_status or "").lower()
    if status in NO_ANSWER_TWILIO_STATUSES:
        return "no-answer"

    combined = " ".join(entry.get("content", "").lower() for entry in transcript_log)
    if any(keyword in combined for keyword in ("voicemail", "voice mail", "mailbox", "leave a message", "at the tone")):
        return "voicemail"
    if any(keyword in combined for keyword in ("reschedule", "another time", "different time", "move the demo", "can't make")):
        return "rescheduled"
    if any(keyword in combined for keyword in ("confirm", "confirmed", "sounds good", "works for me", "see you then")):
        return "confirmed"
    if status == "completed" and transcript_log:
        return "confirmed"
    return "confirmed" if transcript_log else "no-answer"


def extract_action_items(transcript_log: list) -> list:
    """Extract simple action items from assistant responses."""
    action_items = []
    for entry in transcript_log:
        if entry.get("role") != "assistant":
            continue
        text = entry.get("content", "").strip()
        lowered = text.lower()
        if any(token in lowered for token in ("i will", "i'll", "please", "next step", "follow up", "reschedule", "send")):
            action_items.append(text)
    # Deduplicate while preserving order
    return list(dict.fromkeys(action_items))


async def post_outbound_callback(call_sid: str):
    """POST outbound call result to callback URL if configured."""
    context = OUTBOUND_CONTEXTS.get(call_sid, {})
    callback_url = context.get("callback_url")
    result = CALL_RESULTS.get(call_sid)

    if not callback_url or not result or result.get("callback_sent"):
        return

    payload = {
        "call_sid": call_sid,
        "to": result.get("to"),
        "status": result.get("status"),
        "twilio_status": result.get("twilio_status"),
        "transcript": result.get("transcript", []),
        "action_items": result.get("action_items", []),
        "duration": result.get("duration"),
        "completed": result.get("completed", False),
        "updated_at": result.get("updated_at"),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            callback_response = await client.post(callback_url, json=payload)
            callback_response.raise_for_status()
        result["callback_sent"] = True
        result["callback_status_code"] = callback_response.status_code
        logger.info("Posted outbound callback for %s to %s", call_sid, callback_url)
    except Exception as e:
        logger.error("Failed outbound callback for %s: %s", call_sid, e)
        result["callback_error"] = str(e)


async def finalize_outbound_result(call_sid: str, end_reason: str):
    """Finalize outbound result state and trigger callback if needed."""
    if not call_sid:
        return
    if call_sid not in OUTBOUND_CONTEXTS and call_sid not in CALL_RESULTS:
        return

    result = CALL_RESULTS.setdefault(call_sid, {"call_sid": call_sid, "transcript": [], "stream_ended": False})
    twilio_status = (result.get("twilio_status") or "").lower()
    stream_ended = bool(result.get("stream_ended"))
    if not stream_ended or twilio_status not in TERMINAL_TWILIO_STATUSES:
        logger.info(
            "Skipping outbound finalization for %s (stream_ended=%s, twilio_status=%s)",
            call_sid,
            stream_ended,
            twilio_status,
        )
        return

    transcript_log = result.get("transcript", [])
    result["status"] = infer_outbound_status(transcript_log, result.get("twilio_status", ""))
    result["action_items"] = extract_action_items(transcript_log)
    result["completed"] = True
    result["ended_by"] = end_reason
    result["updated_at"] = _utc_now_iso()

    file_path = save_call_result(call_sid, result)
    result["result_file"] = file_path
    await post_outbound_callback(call_sid)


async def create_outbound_call_record(
    request: Request,
    to_number: str,
    resolved_prompt: str,
    prompt_source: str,
    callback_url: str | None = None,
    metadata: dict | None = None,
):
    """Create outbound Twilio call and initialize context/result records."""
    if not twilio_client:
        return Response(
            content=json.dumps({"error": "Twilio not configured"}),
            status_code=500,
            media_type="application/json",
        )

    base_url, ws_base_url = _build_public_and_ws_urls(request)
    if not base_url or not ws_base_url:
        host = request.headers.get("host", "").strip()
        logger.error("No valid PUBLIC_URL/Host for outbound call (public_url=%s host=%s)", PUBLIC_URL, host)
        return Response(
            content=json.dumps({"error": "Server configuration error: no public URL"}),
            status_code=500,
            media_type="application/json",
        )

    path_prefix = os.getenv("PATH_PREFIX", "")
    twiml_url = f"{base_url}{path_prefix}/outbound-twiml"
    status_callback_url = f"{base_url}{path_prefix}/call-status"

    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=twiml_url,
            method="POST",
            status_callback=status_callback_url,
            status_callback_method="POST",
            status_callback_event=["initiated", "ringing", "answered", "completed", "failed", "no-answer", "busy", "canceled"],
        )
    except Exception as e:
        logger.error("Failed to create outbound call: %s", e)
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json",
        )

    OUTBOUND_CONTEXTS[call.sid] = {
        "to": to_number,
        "system_prompt": resolved_prompt,
        "prompt_source": prompt_source,
        "callback_url": callback_url,
        "created_at": _utc_now_iso(),
        "metadata": metadata or {},
    }

    CALL_RESULTS[call.sid] = {
        "call_sid": call.sid,
        "to": to_number,
        "status": "in-progress",
        "twilio_status": call.status,
        "stream_ended": False,
        "transcript": [],
        "action_items": [],
        "duration": None,
        "completed": False,
        "updated_at": _utc_now_iso(),
        "prompt_source": prompt_source,
        "callback_url": callback_url,
        "metadata": metadata or {},
    }

    return Response(
        content=json.dumps(
            {
                "call_sid": call.sid,
                "status": call.status,
                "to": to_number,
                "prompt_source": prompt_source,
                "callback_url": callback_url,
            }
        ),
        media_type="application/json",
    )

# Web search tools definition (DRY with language lookup)
SEARCH_TOOL_DESCRIPTIONS = {
    "en": {
        "description": "Search the internet for current information. Use for: weather, news, business hours, current events, prices, availability, etc.",
        "query_desc": "The search query"
    },
    "de": {
        "description": "Suche im Internet nach aktuellen Informationen. Nutze dies bei Fragen zu: Wetter, Nachrichten, Öffnungszeiten, aktuelle Ereignisse, Preise, Verfügbarkeiten, etc.",
        "query_desc": "Die Suchanfrage"
    }
}

def get_search_tools(language: str) -> list:
    """Get search tools for the given language."""
    desc = SEARCH_TOOL_DESCRIPTIONS.get(language, SEARCH_TOOL_DESCRIPTIONS["en"])
    return [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": desc["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": desc["query_desc"]
                    }
                },
                "required": ["query"]
            }
        }
    }]

SEARCH_TOOLS = get_search_tools(AGENT_LANGUAGE)


async def web_search(query: str) -> str:
    """Search web using Brave API."""
    if not BRAVE_API_KEY:
        logger.error("BRAVE_API_KEY not set, cannot search")
        return "Web-Suche nicht verfügbar. API-Schlüssel fehlt."
    
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": BRAVE_API_KEY,
        "Accept": "application/json"
    }
    # Use module-level MKT_MAP for language-specific market
    mkt = MKT_MAP.get(AGENT_LANGUAGE, "en-US")
    
    params = {
        "q": query,
        "count": 3,
        "offset": 0,
        "mkt": mkt,
        "safesearch": "moderate"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            logger.info(f"Web search: {query}")
            r = await client.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            
            results = data.get("web", {}).get("results", [])
            if not results:
                return "Keine Ergebnisse gefunden."
            
            # Format top 3 results
            summaries = []
            for r in results[:3]:
                title = r.get("title", "")
                desc = r.get("description", "")
                if title and desc:
                    summaries.append(f"{title}: {desc}")
            
            return "\n".join(summaries) if summaries else "Keine relevanten Ergebnisse."
    except httpx.TimeoutException:
        logger.error("Web search timeout")
        return "Suche hat zu lange gedauert."
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Fehler bei der Web-Suche: {str(e)[:50]}"


async def text_to_speech_stream(text: str):
    """Stream TTS from the configured provider (openai or elevenlabs)."""
    if TTS_PROVIDER == "elevenlabs":
        async for chunk in _tts_elevenlabs_stream(text):
            yield chunk
    else:
        async for chunk in _tts_openai_stream(text):
            yield chunk


async def _transcode_mp3_to_mulaw(aiter_bytes):
    """Transcode streamed MP3 bytes to µ-law 8kHz mono for Twilio."""
    ffmpeg = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error", "-hide_banner",
        "-i", "pipe:0",
        "-f", "mulaw", "-ar", "8000", "-ac", "1",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def feed_ffmpeg():
        try:
            async for chunk in aiter_bytes:
                if ffmpeg.stdin is None:
                    break
                ffmpeg.stdin.write(chunk)
                await ffmpeg.stdin.drain()
        finally:
            if ffmpeg.stdin:
                ffmpeg.stdin.close()

    feed_task = asyncio.create_task(feed_ffmpeg())
    try:
        while True:
            if ffmpeg.stdout is None:
                break
            out = await ffmpeg.stdout.read(4096)
            if not out:
                break
            yield out
    finally:
        await feed_task
        rc = await ffmpeg.wait()
        if rc != 0:
            err = b""
            if ffmpeg.stderr:
                err = await ffmpeg.stderr.read()
            logger.error(
                "ffmpeg decode failed (%s): %s",
                rc,
                err.decode("utf-8", errors="replace").strip(),
            )


async def _tts_openai_stream(text: str):
    """Stream TTS from OpenAI API with ffmpeg transcode to µ-law 8kHz."""
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": OPENAI_TTS_MODEL,
        "input": text,
        "voice": OPENAI_TTS_VOICE,
        "response_format": "mp3",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, json=data, headers=headers) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                logger.error(
                    "OpenAI TTS failed (%s): %s",
                    response.status_code,
                    error_body.decode("utf-8", errors="replace").strip(),
                )
                return

            async for chunk in _transcode_mp3_to_mulaw(response.aiter_bytes()):
                yield chunk


async def _tts_elevenlabs_stream(text: str):
    """Stream TTS from ElevenLabs API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream?output_format=ulaw_8000"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    data = {"text": text, "model_id": "eleven_multilingual_v2"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, json=data, headers=headers) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                logger.error(
                    "ElevenLabs TTS failed (%s): %s",
                    response.status_code,
                    error_body.decode("utf-8", errors="replace").strip(),
                )
                return
            content_type = (response.headers.get("content-type") or "").lower()
            if content_type:
                logger.info("ElevenLabs content-type: %s", content_type)

            if "audio/mpeg" in content_type or "audio/mp3" in content_type:
                # ElevenLabs streaming returns MP3 on this plan; decode to mu-law for Twilio.
                async for chunk in _transcode_mp3_to_mulaw(response.aiter_bytes()):
                    yield chunk
                return

            async for chunk in response.aiter_bytes():
                yield chunk


@app.post("/incoming")
@app.post("/v1/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio call."""
    msgs = TWIML_MESSAGES.get(AGENT_LANGUAGE, TWIML_MESSAGES["en"])
    response = VoiceResponse()
    response.say(msgs["connecting"], voice="alice", language=TWIML_LANGUAGE)
    connect = Connect()
    _, ws_base_url = _build_public_and_ws_urls(request)

    if not ws_base_url:
        host = request.headers.get("host", "").strip()
        logger.error("No valid PUBLIC_URL/Host for Twilio stream (public_url=%s host=%s)", PUBLIC_URL, host)
        response.say(msgs["error"], voice="alice", language=TWIML_LANGUAGE)
        response.hangup()
        return Response(content=str(response), media_type="application/xml")

    path_prefix = os.getenv("PATH_PREFIX", "")
    stream_url = f"{ws_base_url}{path_prefix}/stream"
    logger.info("Twilio stream url: %s", stream_url)
    connect.stream(url=stream_url, track="inbound_track")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


@app.post("/outbound")
async def make_outbound_call(request: Request):
    """Initiate outbound call with optional prompt template/callback URL."""
    data = await request.json()
    to_number = str(data.get("to", "")).strip()
    prompt_input = data.get("system_prompt")
    callback_url = data.get("callback_url")

    if not to_number:
        return Response(
            content=json.dumps({"error": "Missing required field: to"}),
            status_code=400,
            media_type="application/json",
        )

    if callback_url and (not isinstance(callback_url, str) or not callback_url.startswith(("http://", "https://"))):
        return Response(
            content=json.dumps({"error": "callback_url must be an http(s) URL"}),
            status_code=400,
            media_type="application/json",
        )

    resolved_prompt, prompt_source = resolve_outbound_prompt(prompt_input)
    return await create_outbound_call_record(
        request=request,
        to_number=to_number,
        resolved_prompt=resolved_prompt,
        prompt_source=prompt_source,
        callback_url=callback_url,
    )


@app.post("/call")
async def make_outbound_call_legacy(request: Request):
    """Backward-compatible outbound endpoint using task prompts."""
    data = await request.json()
    to_number = str(data.get("to", "")).strip()
    task_name = data.get("task", "general")
    task_config = data.get("task_config", {})
    callback_url = data.get("callback_url")

    if not to_number:
        return Response(
            content=json.dumps({"error": "Missing required field: to"}),
            status_code=400,
            media_type="application/json",
        )

    resolved_prompt = get_task_prompt(task_name, task_config) if task_name else SYSTEM_PROMPT
    response = await create_outbound_call_record(
        request=request,
        to_number=to_number,
        resolved_prompt=resolved_prompt,
        prompt_source=f"legacy-task:{task_name}",
        callback_url=callback_url,
        metadata={"task": task_name, "task_config": task_config},
    )

    # Keep legacy map for compatibility with existing /calls payload.
    if response.status_code < 400:
        payload = json.loads(response.body.decode("utf-8"))
        active_calls[payload["call_sid"]] = {
            "to": to_number,
            "task": task_name,
            "task_config": task_config,
            "started": _utc_now_iso(),
        }
    return response


@app.post("/outbound-twiml")
async def outbound_twiml(request: Request):
    """Return TwiML for outbound calls and connect to websocket audio bridge."""
    data = await request.form()
    call_sid = data.get("CallSid")
    msgs = TWIML_MESSAGES.get(AGENT_LANGUAGE, TWIML_MESSAGES["en"])
    response = VoiceResponse()
    response.say(msgs["greeting"], voice="alice", language=TWIML_LANGUAGE)
    connect = Connect()

    _, ws_base_url = _build_public_and_ws_urls(request)
    if not ws_base_url:
        host = request.headers.get("host", "").strip()
        logger.error("No valid PUBLIC_URL/Host for outbound stream (public_url=%s host=%s)", PUBLIC_URL, host)
        response.say(msgs["error"], voice="alice", language=TWIML_LANGUAGE)
        response.hangup()
        return Response(content=str(response), media_type="application/xml")

    path_prefix = os.getenv("PATH_PREFIX", "")
    stream_url = f"{ws_base_url}{path_prefix}/stream"
    if call_sid:
        stream_url = f"{stream_url}?outbound_sid={call_sid}"
    logger.info("Outbound stream url: %s", stream_url)
    connect.stream(url=stream_url, track="inbound_track")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


@app.post("/call-status")
async def handle_call_status(request: Request):
    """Handle call status callbacks from Twilio."""
    data = await request.form()
    call_sid = data.get("CallSid")
    status = data.get("CallStatus")
    duration = data.get("CallDuration", 0)
    
    logger.info(f"Call {call_sid} status: {status}, duration: {duration}s")

    if call_sid and call_sid in CALL_RESULTS:
        result = CALL_RESULTS[call_sid]
        result["twilio_status"] = status
        result["duration"] = duration
        result["updated_at"] = _utc_now_iso()
        status_lower = (status or "").lower()
        if status_lower in TERMINAL_TWILIO_STATUSES and result.get("stream_ended"):
            await finalize_outbound_result(call_sid, "status-webhook")

    if call_sid in active_calls:
        call_info = active_calls[call_sid]
        call_info["status"] = status
        call_info["duration"] = duration

        # Save to results
        call_results[call_sid] = call_info

        # Clean up active
        if status in TERMINAL_TWILIO_STATUSES:
            del active_calls[call_sid]
    
    return Response(content="OK", media_type="text/plain")


@app.get("/outbound-result/{call_sid}")
async def get_outbound_result(call_sid: str):
    """Get outbound call result by call SID."""
    result = CALL_RESULTS.get(call_sid)
    if not result:
        return Response(
            content=json.dumps({"error": f"No outbound result found for {call_sid}"}),
            status_code=404,
            media_type="application/json",
        )
    return result


@app.get("/calls")
async def list_calls():
    """List all call results."""
    return {
        "active": active_calls,
        "completed": call_results,
        "outbound_contexts": OUTBOUND_CONTEXTS,
        "outbound_results": CALL_RESULTS,
    }


@app.websocket("/stream")
@app.websocket("/v1/stream")
async def websocket_endpoint(twilio_ws: WebSocket):
    """Handle Twilio WebSocket and bridge to Deepgram."""
    await twilio_ws.accept()
    
    # Query params can include legacy task and outbound call SID context.
    task_name = twilio_ws.query_params.get("task", "")
    outbound_sid = twilio_ws.query_params.get("outbound_sid", "").strip()
    call_sid = outbound_sid or None
    task_config = None
    transcript_log = []
    
    if outbound_sid:
        logger.info("WebSocket accepted for outbound call: %s", outbound_sid)
    elif task_name:
        logger.info(f"WebSocket accepted for task: {task_name}")
    else:
        logger.info("WebSocket accepted for inbound call")
    
    # Initialize prompt. Outbound calls can override this with stored context.
    system_prompt = SYSTEM_PROMPT
    if outbound_sid and outbound_sid in OUTBOUND_CONTEXTS:
        context = OUTBOUND_CONTEXTS[outbound_sid]
        system_prompt = context.get("system_prompt") or SYSTEM_PROMPT
        logger.info("Loaded outbound prompt for call %s from %s", outbound_sid, context.get("prompt_source"))

        result = CALL_RESULTS.setdefault(
            outbound_sid,
            {"call_sid": outbound_sid, "transcript": [], "stream_ended": False},
        )
        result["to"] = context.get("to")
        result["callback_url"] = context.get("callback_url")
        result["prompt_source"] = context.get("prompt_source")
        result["updated_at"] = _utc_now_iso()

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    conversation_history = [{"role": "system", "content": system_prompt}]

    stream_sid = None
    audio_queue = asyncio.Queue(maxsize=AUDIO_QUEUE_SIZE)
    stop_event = asyncio.Event()
    dg_ready = asyncio.Event()
    tts_playing = asyncio.Event()

    # Deepgram WebSocket URL (language from config)
    dg_language = "de" if AGENT_LANGUAGE == "de" else "en-US"
    dg_url = (
        f"wss://api.deepgram.com/v1/listen?"
        f"encoding=mulaw&sample_rate=8000&channels=1"
        f"&model=nova-2&language={dg_language}&punctuate=true"
        f"&interim_results=true&utterance_end=400&endpointing=300&vad_events=true"
    )
    dg_headers = [("Authorization", f"Token {DEEPGRAM_API_KEY}")]

    async def play_typing_sound(stop_typing: asyncio.Event):
        """Play typing sound in a loop until stop_typing is set."""
        if not TYPING_SOUND_DATA or not stream_sid:
            return
        try:
            while not stop_typing.is_set() and not stop_event.is_set():
                # Send typing sound in frames
                for i in range(0, len(TYPING_SOUND_DATA), TWILIO_FRAME_BYTES):
                    if stop_typing.is_set() or stop_event.is_set():
                        break
                    frame = TYPING_SOUND_DATA[i:i + TWILIO_FRAME_BYTES]
                    if len(frame) < TWILIO_FRAME_BYTES:
                        frame = frame + b'\xff' * (TWILIO_FRAME_BYTES - len(frame))
                    payload = base64.b64encode(frame).decode("utf-8")
                    await twilio_ws.send_json({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload}
                    })
                    await asyncio.sleep(TWILIO_FRAME_DELAY_SEC)
        except Exception as e:
            logger.debug(f"Typing sound stopped: {e}")

    async def process_transcript(transcript: str):
        """Process transcript and generate response."""
        nonlocal stream_sid
        if not transcript.strip():
            return

        logger.info(f"User said: {transcript}")
        transcript_log.append({"role": "user", "content": transcript, "timestamp": asyncio.get_event_loop().time()})
        conversation_history.append({"role": "user", "content": transcript})
        if call_sid and call_sid in CALL_RESULTS:
            CALL_RESULTS[call_sid]["transcript"] = list(transcript_log)
            CALL_RESULTS[call_sid]["updated_at"] = _utc_now_iso()

        # Limit conversation history to avoid exceeding model context
        if len(conversation_history) > MAX_HISTORY:
            conversation_history[:] = [conversation_history[0]] + conversation_history[-(MAX_HISTORY-1):]

        # Start typing sound while waiting for AI
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(play_typing_sound(stop_typing))

        try:
            completion = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                max_tokens=120
            )
            ai_text = completion.choices[0].message.content or "Das System verarbeitet Ihre Anfrage."

            # Stop typing sound before TTS
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            logger.info(f"AI response: {ai_text}")
            transcript_log.append({"role": "assistant", "content": ai_text, "timestamp": asyncio.get_event_loop().time()})
            conversation_history.append({"role": "assistant", "content": ai_text})

            if call_sid and call_sid in CALL_RESULTS:
                CALL_RESULTS[call_sid]["transcript"] = list(transcript_log)
                CALL_RESULTS[call_sid]["updated_at"] = _utc_now_iso()

            if stream_sid:
                tts_playing.set()
                try:
                    buffer = b""
                    frames_sent = 0
                    async for audio_chunk in text_to_speech_stream(ai_text):
                        if stop_event.is_set():
                            break
                        buffer += audio_chunk
                        # Prevent buffer overflow
                        if len(buffer) > MAX_TTS_BUFFER:
                            logger.warning("TTS buffer overflow, dropping old audio")
                            buffer = buffer[-MAX_TTS_BUFFER:]
                        while len(buffer) >= TWILIO_FRAME_BYTES:
                            frame, buffer = (
                                buffer[:TWILIO_FRAME_BYTES],
                                buffer[TWILIO_FRAME_BYTES:],
                            )
                            payload = base64.b64encode(frame).decode("utf-8")
                            await twilio_ws.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": payload}
                            })
                            frames_sent += 1
                            if frames_sent % 200 == 0:
                                logger.info("Sent %s TTS frames to Twilio", frames_sent)
                            await asyncio.sleep(TWILIO_FRAME_DELAY_SEC)
                finally:
                    tts_playing.clear()
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
        finally:
            # Ensure typing sound stops even on error
            stop_typing.set()
            typing_task.cancel()

    async def twilio_receiver():
        """Receive messages from Twilio."""
        nonlocal stream_sid, call_sid, task_name, task_config, system_prompt
        try:
            while not stop_event.is_set():
                message = await twilio_ws.receive_text()
                data = json.loads(message)

                if data['event'] == 'connected':
                    logger.info("Twilio: connected")
                elif data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    twilio_call_sid = data['start'].get('callSid')
                    if twilio_call_sid:
                        call_sid = twilio_call_sid
                    logger.info(f"Twilio: stream started (stream={stream_sid}, call={call_sid})")

                    # Load outbound prompt context if this is an outbound call.
                    if call_sid and call_sid in OUTBOUND_CONTEXTS:
                        context = OUTBOUND_CONTEXTS[call_sid]
                        system_prompt = context.get("system_prompt") or SYSTEM_PROMPT
                        conversation_history[0] = {"role": "system", "content": system_prompt}
                        result = CALL_RESULTS.setdefault(
                            call_sid,
                            {"call_sid": call_sid, "transcript": [], "stream_ended": False},
                        )
                        result["to"] = context.get("to")
                        result["prompt_source"] = context.get("prompt_source")
                        result["callback_url"] = context.get("callback_url")
                        result["updated_at"] = _utc_now_iso()
                        logger.info("Applied outbound context for %s", call_sid)

                    # Look up task info using call_sid
                    if call_sid and call_sid in active_calls:
                        call_info = active_calls[call_sid]
                        task_name = call_info.get("task", task_name)
                        task_config = call_info.get("task_config", {})
                        
                        # Update system prompt with task config
                        if task_name:
                            system_prompt = get_task_prompt(task_name, task_config)
                            conversation_history[0] = {"role": "system", "content": system_prompt}
                            logger.info(f"Loaded task '{task_name}' with config: {task_config}")
                elif data['event'] == 'media':
                    # Only process inbound audio (caller's voice), not outbound (TTS)
                    track = data.get('media', {}).get('track', 'inbound')
                    if track in ('inbound', 'inbound_track'):
                        audio = base64.b64decode(data['media']['payload'])
                        try:
                            audio_queue.put_nowait(audio)
                        except asyncio.QueueFull:
                            pass  # Drop audio if queue is full
                elif data['event'] == 'stop':
                    logger.info("Twilio: stream stopped")
                    stop_event.set()
                    break
        except WebSocketDisconnect:
            logger.info("Twilio: disconnected")
            stop_event.set()
        except Exception as e:
            logger.error(f"Twilio receiver error: {e}")
            stop_event.set()

    async def deepgram_sender(dg_ws):
        """Send audio from queue to Deepgram."""
        try:
            await asyncio.wait_for(dg_ready.wait(), timeout=DEEPGRAM_CONNECT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("Deepgram connection timeout")
            stop_event.set()
            return

        logger.info("Deepgram sender: starting")
        audio_count = 0
        try:
            while not stop_event.is_set():
                try:
                    audio = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    await dg_ws.send(audio)
                    audio_count += 1
                    if audio_count % 100 == 0:
                        logger.info(f"Sent {audio_count} audio chunks to Deepgram")
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.error("Deepgram connection closed")
                    stop_event.set()
                    break
        except Exception as e:
            logger.error(f"Deepgram sender error: {e}")

    async def deepgram_receiver(dg_ws):
        """Receive transcripts from Deepgram."""
        try:
            async for message in dg_ws:
                if stop_event.is_set():
                    break
                data = json.loads(message)
                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")
                        is_final = data.get("is_final", False)
                        speech_final = data.get("speech_final", False)
                        if transcript:
                            logger.info(f"Deepgram: '{transcript}' (final={is_final}, speech_final={speech_final})")
                        # Only act on speech_final to avoid duplicate replies
                        # Skip if TTS is playing to prevent response overlap
                        if speech_final and transcript and not tts_playing.is_set():
                            await process_transcript(transcript)
        except Exception as e:
            logger.error(f"Deepgram receiver error: {e}")

    try:
        # Start Twilio receiver immediately (buffers audio)
        twilio_task = asyncio.create_task(twilio_receiver())

        # Wait for stream_sid
        for _ in range(50):
            if stream_sid:
                break
            await asyncio.sleep(0.1)

        if not stream_sid:
            logger.error("No stream_sid received")
            return

        logger.info("Connecting to Deepgram...")

        async with websockets.connect(dg_url, additional_headers=dg_headers) as dg_ws:
            logger.info("Deepgram: connected")
            dg_ready.set()

            # Run sender and receiver
            sender_task = asyncio.create_task(deepgram_sender(dg_ws))
            receiver_task = asyncio.create_task(deepgram_receiver(dg_ws))

            # Wait for stop
            await stop_event.wait()

            sender_task.cancel()
            receiver_task.cancel()

        twilio_task.cancel()
        logger.info("Session ended")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if call_sid and (call_sid in OUTBOUND_CONTEXTS or call_sid in CALL_RESULTS):
            result = CALL_RESULTS.setdefault(
                call_sid,
                {"call_sid": call_sid, "transcript": [], "stream_ended": False},
            )
            result["stream_ended"] = True
            result["transcript"] = list(transcript_log)
            result["conversation"] = conversation_history[1:]  # Skip system prompt
            result["updated_at"] = _utc_now_iso()
            await finalize_outbound_result(call_sid, "websocket-disconnect")
        elif call_sid and transcript_log:
            # Legacy behavior for non-outbound tracked calls.
            result = {
                "call_sid": call_sid,
                "task": task_name,
                "transcript": transcript_log,
                "conversation": conversation_history[1:],  # Skip system prompt
                "completed": True,
            }
            file_path = save_call_result(call_sid, result)
            logger.info(f"Call transcript saved to {file_path}")


if __name__ == "__main__":
    tts_info = (
        f"OpenAI ({OPENAI_TTS_MODEL}, voice={OPENAI_TTS_VOICE})"
        if TTS_PROVIDER != "elevenlabs"
        else f"ElevenLabs (voice_id={ELEVENLABS_VOICE_ID})"
    )
    logger.info("TTS provider: %s", tts_info)
    uvicorn.run(app, host=HOST, port=PORT)
