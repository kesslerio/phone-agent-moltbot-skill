# Phone Agent Moltbot Skill

A real-time AI voice agent that handles incoming phone calls using Twilio, transcribes speech with Deepgram, generates responses via OpenAI, and speaks back with OpenAI text-to-speech by default (with optional ElevenLabs TTS).

## Features

- **Real-time Voice Processing**: Handles incoming Twilio calls with low-latency WebSocket audio
- **Automatic Speech Recognition**: Deepgram for fast, accurate transcription
- **AI-Powered Responses**: OpenAI GPT for intelligent conversation
- **Natural Speech Output**: OpenAI TTS by default, with optional ElevenLabs streaming TTS
- **Task-Based Automation**: Configurable task definitions for specific agent behaviors
- **Recording & Logging**: Automatic call recording and conversation logs

## Architecture

```
Incoming Call (Twilio Phone)
         |
         v
  Twilio WebSocket (Audio Stream)
         |
         +---> Local FastAPI Server
         |           |
         |           +---> Deepgram (Speech-to-Text)
         |           |
         |           +---> OpenAI (LLM/Intelligence)
         |           |
         |           +---> OpenAI TTS (default) / ElevenLabs TTS (optional)
         |           |
         +---------- (Audio Response)
         |
    Phone Speaker Output
```

## Prerequisites

Before you begin, ensure you have:

1. **Twilio Account**
   - Active Twilio account with a phone number
   - TwiML App configured
   - Account SID and Auth Token

2. **API Keys** (free tier available for all)
   - Deepgram API Key (https://console.deepgram.com/)
   - OpenAI API Key (https://platform.openai.com/api-keys)
   - ElevenLabs API Key (optional; only needed when `TTS_PROVIDER=elevenlabs`) (https://elevenlabs.io/)

3. **Local Network Access**
   - Ngrok or similar tool to expose localhost to the internet
   - Ability to accept incoming webhooks from Twilio

4. **Python 3.9+** and pip

## Installation

```bash
# Clone the repository
git clone https://github.com/kesslerio/phone-agent-moltbot-skill.git
cd phone-agent-moltbot-skill

# Install dependencies
pip install -r scripts/requirements.txt
```

## Configuration

### Set Environment Variables

Create a `.env` file or set environment variables:

```bash
# API Keys (required)
export DEEPGRAM_API_KEY="your-deepgram-key"
export OPENAI_API_KEY="your-openai-key"

# TTS Provider (optional; default is OpenAI)
export TTS_PROVIDER="openai"  # "openai" (default) or "elevenlabs"
export OPENAI_TTS_VOICE="echo"
export OPENAI_TTS_MODEL="tts-1"

# ElevenLabs (optional; only when TTS_PROVIDER=elevenlabs)
export ELEVENLABS_API_KEY="your-elevenlabs-key"

# Twilio (required)
export TWILIO_ACCOUNT_SID="your-account-sid"
export TWILIO_AUTH_TOKEN="your-auth-token"
export TWILIO_PHONE_NUMBER="+18665515246"  # Your Twilio number

# Server (optional)
export PORT=8080
export PUBLIC_URL="https://your-ngrok-url.ngrok.io"  # For webhooks

# Voice Customization (optional)
export ELEVENLABS_VOICE_ID="onwK4e9ZLuTAKqWW03F9"  # only for ElevenLabs provider

# System Prompt Configuration (optional)
export SYSTEM_PROMPT_FILE="/path/to/custom-prompt.txt"  # Load prompt from file
export SYSTEM_PROMPT_FILE_REQUIRED="true"  # Exit if file missing (default: false)
export SYSTEM_PROMPT="Custom prompt text here"  # Override built-in prompt
```

**Template Variables:** When using `SYSTEM_PROMPT_FILE`, you can include these placeholders:
- `{agent_name}` - Replaced with `AGENT_NAME` env var (default: "Assistant")
- `{owner_name}` - Replaced with `OWNER_NAME` env var (default: "your")
- `{language}` - Replaced with `AGENT_LANGUAGE` env var (default: "en")

Example custom prompt file:
```
You are {agent_name}, {owner_name} personal assistant.
Speak in {language} with precision and clarity.
```

Or add to `~/.moltbot/.env` or `~/.clawdbot/.env`:

```
DEEPGRAM_API_KEY=your-key
OPENAI_API_KEY=your-key
TTS_PROVIDER=openai
ELEVENLABS_API_KEY=your-key
TWILIO_ACCOUNT_SID=your-sid
TWILIO_AUTH_TOKEN=your-token
TWILIO_PHONE_NUMBER=+1...
```

## Startup & Configuration

### 1. Start the Local Server

```bash
python3 scripts/server.py
```

The server will start on `http://localhost:8080` by default.

### 2. Expose to Internet with Ngrok

In another terminal:

```bash
ngrok http 8080
```

Note the HTTPS URL (e.g., `https://abc123.ngrok.io`)

### 3. Configure Twilio Webhook

In Twilio Console:

1. Go to **Phone Numbers** → Your number
2. Under **Voice & Fax**:
   - Set "A Call Comes In" to **Webhook**
   - URL: `https://<your-ngrok-url>.ngrok.io/incoming`
   - Method: `POST`
3. Save

### 4. Test Incoming Calls

Call your Twilio number. The agent will:
1. Answer and greet you
2. Listen to your speech
3. Transcribe your words
4. Generate a response via OpenAI
5. Speak the response back to you

## Customization

### Change Agent Persona

Edit `SYSTEM_PROMPT` in `scripts/server.py`:

```python
SYSTEM_PROMPT = """You are a helpful customer service agent. Be friendly, concise, and professional."""
```

### Change Voice

OpenAI is the default TTS provider:

```bash
export TTS_PROVIDER="openai"
export OPENAI_TTS_VOICE="echo"
```

To use ElevenLabs instead:

```bash
export TTS_PROVIDER="elevenlabs"
export ELEVENLABS_VOICE_ID="g1r0eKKcGkk7Ep0RVcVn"  # Callum voice
```

Available ElevenLabs voices: https://elevenlabs.io/docs/getting-started/voices

### Use Different Model

Edit `scripts/server.py` and change the OpenAI model:

```python
response = await client.chat.completions.create(
    model="gpt-4",  # or "gpt-4-turbo" for faster responses
    messages=messages,
)
```

### Task-Based Behaviors

Create YAML task definitions in the `tasks/` directory:

```yaml
name: book_restaurant
description: "Help the user book a restaurant reservation"
system_prompt: "You are a friendly restaurant reservation assistant..."
actions:
  - confirm_date
  - confirm_time
  - confirm_party_size
  - book_reservation
```

## Integration with Moltbot

Add this skill to your Moltbot configuration:

```json
{
  "skills": [
    {
      "name": "phone-agent",
      "path": "/path/to/phone-agent-moltbot-skill",
      "enabled": true
    }
  ]
}
```

Then reference it in workflows:
- "Set up an incoming voice agent"
- "Configure a customer service chatbot"
- "Test voice AI capabilities"

## Project Structure

```
phone-agent-moltbot-skill/
├── scripts/
│   ├── server.py              # Main FastAPI server
│   ├── server_realtime.py     # Realtime processing variant
│   ├── requirements.txt       # Python dependencies
│   └── typing_sound.raw       # Typing sound effect
├── tasks/
│   ├── book_restaurant.yaml   # Example task definitions
│   └── get_quote.yaml         # Example task definitions
├── calls/                     # Recording storage directory
├── references/                # Supporting documentation
├── SKILL.md                   # Moltbot skill manifest
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## Troubleshooting

### Server Won't Start

- Check Python version: `python3 --version` (requires 3.9+)
- Install dependencies: `pip install -r scripts/requirements.txt`
- Check PORT variable: `echo $PORT` (should be 8080 or set value)

### Twilio Webhook Not Connecting

- Verify ngrok is running and the URL matches your Twilio webhook
- Check server logs: `python3 scripts/server.py` (should show incoming requests)
- Test ngrok tunnel: `curl https://<your-ngrok-url>.ngrok.io/health`

### Poor Transcription Quality

- Ensure DEEPGRAM_API_KEY is valid
- Check microphone/audio quality on the calling phone
- Deepgram is very accurate; poor results indicate audio issues

### Slow Responses

- OpenAI API latency varies; gpt-4o-mini is fast and cheap
- Switch to "gpt-3.5-turbo" for faster responses (less capable)
- Increase timeout in websocket settings if needed

### Voice Not Speaking

- Verify `TTS_PROVIDER` is set correctly (`openai` by default, `elevenlabs` for ElevenLabs)
- If using OpenAI TTS, verify `OPENAI_API_KEY` is valid
- If using ElevenLabs TTS, verify `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` are valid
- Confirm audio is not muted on the receiving phone

## API Reference

### Incoming Call Webhook

**POST** `/incoming`

Twilio sends call information to this endpoint. The server responds with TwiML to establish WebSocket connection.

### WebSocket Audio Stream

**WS** `/ws`

Bidirectional audio stream for incoming call processing.

### Health Check

**GET** `/health`

Returns `{"status": "ok"}` if the server is running.

## Performance & Scaling

Current implementation handles:
- Single concurrent call per server instance
- ~100ms RTT for transcription + LLM + TTS
- Suitable for demo/testing, hobby projects, and low-volume use

For production:
- Run multiple server instances behind a load balancer
- Use Twilio's call queuing
- Implement connection pooling for API clients
- Consider dedicated hardware for Deepgram and your selected TTS provider (OpenAI by default, or ElevenLabs)

## Deployment Options

### Local Development
```bash
python3 scripts/server.py
ngrok http 8080
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/ .
CMD ["python3", "server.py"]
```

Build and run:
```bash
docker build -t phone-agent .
docker run -p 8080:8080 \
  -e DEEPGRAM_API_KEY="..." \
  -e OPENAI_API_KEY="..." \
  -e TTS_PROVIDER="openai" \
  -e ELEVENLABS_API_KEY="..." \
  -e TWILIO_ACCOUNT_SID="..." \
  -e TWILIO_AUTH_TOKEN="..." \
  phone-agent
```

`ELEVENLABS_API_KEY` is optional unless `TTS_PROVIDER=elevenlabs`.

### Cloud Deployment

- **Heroku**: Add `Procfile` → `web: python3 scripts/server.py`
- **Railway.app**: Auto-detects Python and builds
- **AWS Lambda**: Use WebSocket API Gateway + Lambda
- **Google Cloud Run**: Containerize and deploy

## License

MIT

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request

## Support

- MCP Server / APIs: [Deepgram](https://deepgram.com/) | [OpenAI](https://openai.com/) (LLM + default TTS) | [ElevenLabs](https://elevenlabs.io/) (optional TTS)
- Twilio Docs: [Voice API](https://www.twilio.com/docs/voice)
- Moltbot: [Documentation](https://moltbot.io/)

## Requirements

- `ffmpeg` must be in PATH (for converting OpenAI/ElevenLabs MP3 audio to Twilio mu-law audio)
- If running as a systemd service, ensure PATH includes ffmpeg location:
  ```ini
  Environment=PATH=/home/art/.nix-profile/bin:/usr/bin:/bin
  ```
