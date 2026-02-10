---
name: phone-agent
description: "Run a real-time AI phone agent using Twilio, Deepgram, and OpenAI/ElevenLabs TTS. Handles incoming calls, transcribes audio, generates responses via LLM, and speaks back via streaming TTS. Use when user wants to: (1) Test voice AI capabilities, (2) Handle phone calls programmatically, (3) Build a conversational voice bot."
---

# Phone Agent Skill

Runs a local FastAPI server that acts as a real-time voice bridge.

## Architecture

```
Twilio (Phone) <--> WebSocket (Audio) <--> [Local Server] <--> Deepgram (STT)
                                                  |
                                                  +--> OpenAI (LLM)
                                                  +--> OpenAI TTS or ElevenLabs (TTS)
```

## Prerequisites

1.  **Twilio Account**: Phone number + TwiML App.
2.  **Deepgram API Key**: For fast speech-to-text.
3.  **OpenAI API Key**: For conversation logic + TTS (default).
4.  **ElevenLabs API Key** (optional): For higher-quality TTS (set `TTS_PROVIDER=elevenlabs`).
5.  **Ngrok** (or similar): To expose your local port 8080 to Twilio.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r scripts/requirements.txt
    ```

2.  **Set Environment Variables** (in `~/.moltbot/.env`, `~/.clawdbot/.env`, or export):
    ```bash
    export DEEPGRAM_API_KEY="your_key"
    export OPENAI_API_KEY="your_key"
    export TWILIO_ACCOUNT_SID="your_sid"
    export TWILIO_AUTH_TOKEN="your_token"
    export PORT=8080

    # TTS Provider (default: openai — ~6x cheaper than ElevenLabs)
    export TTS_PROVIDER="openai"          # or "elevenlabs"
    export OPENAI_TTS_VOICE="echo"        # alloy, echo, fable, onyx, nova, shimmer
    export OPENAI_TTS_MODEL="tts-1"       # tts-1 (fast) or tts-1-hd (quality)

    # Only needed if TTS_PROVIDER=elevenlabs
    export ELEVENLABS_API_KEY="your_key"
    export ELEVENLABS_VOICE_ID="onwK4e9ZLuTAKqWW03F9"
    ```

    **Optional - System Prompt Customization** (priority: file > env var > built-in):
    ```bash
    # Option 1: Load from file
    export SYSTEM_PROMPT_FILE="/path/to/custom-prompt.txt"
    
    # Option 2: Set directly via env var
    export SYSTEM_PROMPT="You are a helpful phone assistant. Be concise and friendly."
    
    # Option 3: Use built-in defaults with name customization
    export AGENT_NAME="Niemand"
    export OWNER_NAME="Martin's"
    ```

3.  **Start the Server**:
    ```bash
    python3 scripts/server.py
    ```

4.  **Expose to Internet**:
    ```bash
    ngrok http 8080
    ```

5.  **Configure Twilio**:
    - Go to your Phone Number settings.
    - Set "Voice & Fax" -> "A Call Comes In" to **Webhook**.
    - URL: `https://<your-ngrok-url>.ngrok.io/incoming`
    - Method: `POST`

## Usage

Call your Twilio number. The agent should answer, transcribe your speech, think, and reply in a natural voice.

## Customization

- **System Prompt**: Configure via `SYSTEM_PROMPT_FILE` (load from file), `SYSTEM_PROMPT` (env var), or modify the built-in defaults with `AGENT_NAME` and `OWNER_NAME`.
- **TTS Provider**: Set `TTS_PROVIDER=openai` (default, $0.03/min) or `TTS_PROVIDER=elevenlabs` ($0.17/min, higher quality).
- **Voice (OpenAI)**: Set `OPENAI_TTS_VOICE` — options: alloy, echo, fable, onyx, nova, shimmer.
- **Voice (ElevenLabs)**: Change `ELEVENLABS_VOICE_ID` to use different voices.
- **Model**: Switch `gpt-4o-mini` to `gpt-4` for smarter (but slower) responses.
- **Language**: Set `AGENT_LANGUAGE` to `en` or `de` for English or German.
