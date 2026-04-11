# Self-Hosted ChatGPT: Open WebUI + EXO on DGX Spark

**Goal:** Put a polished chat interface in front of your EXO cluster so anyone on your network can use your local AI models — no cloud, no API keys, no per-token fees.

```
 ┌───────────────────┐     ┌───────────────────────────┐     ┌───────────────────┐
 │  Browser           │     │  Open WebUI                │     │  EXO Cluster       │
 │  (any device)      │ ──► │  http://192.168.0.188:3000 │ ──► │  :52415 (API)      │
 │                    │     │  Chat UI, user accounts,   │     │  Llama, Qwen,      │
 │                    │     │  history, RAG, presets      │     │  DeepSeek, etc.    │
 └───────────────────┘     └───────────────────────────┘     └───────────────────┘
```

---

## Why Open WebUI?

EXO already has TinyChat (a basic web UI), but Open WebUI gives you:

| Feature | TinyChat (EXO) | Open WebUI |
|---|---|---|
| Chat interface | Basic | Full-featured (GPT-like) |
| User accounts | No | Yes (multi-user, roles) |
| Chat history | No | Yes (persistent, searchable) |
| RAG (upload docs) | No | Yes (built-in) |
| Model switching | Yes | Yes (dropdown) |
| System prompts / presets | No | Yes |
| Mobile-friendly | Partial | Yes |
| API key management | No | Yes |

---

## Setup

### Option 1: Docker (Recommended)

```bash
# On spark-dgx-1 (or any machine that can reach the EXO API)

docker run -d \
    --name open-webui \
    --restart always \
    -p 3000:8080 \
    -e OPENAI_API_BASE_URL=http://10.0.0.1:52415/v1 \
    -e OPENAI_API_KEY=not-needed \
    -e WEBUI_AUTH=true \
    -v open-webui-data:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

Open `http://192.168.0.188:3000` in your browser. Create an admin account on first visit.

### Option 2: pip Install

```bash
pip install open-webui

# Set the EXO API endpoint
export OPENAI_API_BASE_URL=http://10.0.0.1:52415/v1
export OPENAI_API_KEY=not-needed

# Start
open-webui serve --host 0.0.0.0 --port 3000
```

---

## Configuration

### Point Open WebUI at EXO

In the Open WebUI admin panel (`Settings → Connections`):

1. **API Base URL:** `http://10.0.0.1:52415/v1`
2. **API Key:** `not-needed` (EXO doesn't require auth)
3. Click **Verify** — should show your available models

### Model Presets

Create presets for common tasks in `Settings → Models`:

| Preset Name | Model | System Prompt | Use Case |
|---|---|---|---|
| General Assistant | `llama-3.1-70b` | "You are a helpful assistant." | General chat |
| Code Helper | `qwen-2.5-72b-instruct` | "You are an expert programmer." | Development |
| Document Drafter | `llama-3.1-8b` | "You are a professional writer." | Business docs |

### User Management

Open WebUI supports multiple users with role-based access:

- **Admin:** Full access, manage models and settings
- **User:** Chat only, can't change settings
- **Pending:** New signups waiting for admin approval

Set `WEBUI_AUTH=true` (default) to require login.

---

## Adding LiteLLM as a Router (Optional)

If you want to route between your local EXO models and cloud APIs (OpenAI, Anthropic) as a fallback:

```bash
pip install litellm

# Create config
cat > litellm_config.yaml << 'EOF'
model_list:
  # Local models via EXO
  - model_name: llama-3.1-70b
    litellm_params:
      model: openai/llama-3.1-70b
      api_base: http://10.0.0.1:52415/v1
      api_key: not-needed

  - model_name: llama-3.2-3b
    litellm_params:
      model: openai/llama-3.2-3b
      api_base: http://10.0.0.1:52415/v1
      api_key: not-needed

  # Cloud fallback (optional)
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
EOF

# Start LiteLLM proxy
litellm --config litellm_config.yaml --host 0.0.0.0 --port 4000
```

Then point Open WebUI at LiteLLM (`http://10.0.0.1:4000/v1`) instead of EXO directly. Users can switch between local and cloud models in the same interface.

---

## Systemd Service

Run Open WebUI as a persistent service:

```bash
sudo tee /etc/systemd/system/open-webui.service << 'EOF'
[Unit]
Description=Open WebUI
After=network-online.target exo-cuda.service
Wants=network-online.target

[Service]
Type=simple
User=sem
Environment="OPENAI_API_BASE_URL=http://10.0.0.1:52415/v1"
Environment="OPENAI_API_KEY=not-needed"
ExecStart=/usr/local/bin/open-webui serve --host 0.0.0.0 --port 3000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable open-webui
sudo systemctl start open-webui
```

---

## Access Points

| Service | URL | Purpose |
|---|---|---|
| Open WebUI | `http://192.168.0.188:3000` | Chat interface (users go here) |
| EXO TinyChat | `http://192.168.0.188:52415` | Basic chat + API |
| EXO API | `http://192.168.0.188:52415/v1` | OpenAI-compatible API |
| LiteLLM (optional) | `http://192.168.0.188:4000` | Multi-model router |

---

## References

- [Open WebUI](https://github.com/open-webui/open-webui)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [EXO CUDA/Tinygrad Setup](07-exo-cuda-tinygrad.md)
