# PhantomFlow Voice Intelligence Backend (`backendy`)

Production backend services for PhantomFlow — the sub-second duplex voice intelligence engine tailored for Indian English and Hinglish.

---

## 🛠️ Architecture

The backend consists of two integrated services:

1. **Python FastAPI Core Voice Orchestrator** (`src/phantomflow/` & `speech/`)
   - Bidirectional duplex audio streaming over WebSocket (`/ws/voice`)
   - Sub-800ms pipeline orchestration (Silero VAD + streaming ASR + TTS)
   - GhostBarge turn cancellation & latency profiling
   - In-domain synthetic benchmark evaluation suite

2. **Node.js Express Conversation Engine** (`backend/`)
   - Intent resolution & dual-path routing (Fast Lane deterministic vs Smart Path LLM)
   - SwitchSync entity preservation across mid-sentence language switches
   - Railway domain integration (PNR status, live train status, train search)
   - MongoDB Atlas session persistence

---

## 🚀 Deploy to Render

This repository includes a [`render.yaml`](./render.yaml) Blueprint configuration.

### Option 1: Automatic Blueprint (Recommended)
1. In Render Dashboard, click **New +** → **Blueprint**.
2. Connect `https://github.com/uservinay777-space/backendy`.
3. Render automatically provisions:
   - `phantomflow-conversation-engine` (Node Web Service)
   - `phantomflow-orchestrator` (Python Web Service)
4. Add your environment variables in the Render dashboard:
   - `GROQ_API_KEY`: Groq API key for Smart Path LLM
   - `SARVAM_API_KEY`: Sarvam AI subscription key
   - `MONGODB_URI`: MongoDB Atlas connection string (optional)

### Option 2: Deploy as Node Web Service
- **Build Command**: `npm install && npm --prefix backend install`
- **Start Command**: `node backend/src/server.js`
- **Port**: Render automatically injects `PORT` (e.g. `10000`)

### Option 3: Deploy as Python Web Service
- **Build Command**: `pip install -r requirements.txt websockets uvicorn fastapi`
- **Start Command**: `PYTHONPATH=src uvicorn phantomflow.server:app --host 0.0.0.0 --port $PORT`

---

## 💻 Local Development

### 1. Node.js Conversation Service (Port 5001)
```bash
cd backend
npm install
npm run dev
```

### 2. Python Orchestrator Service (Port 8000)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src uvicorn phantomflow.server:app --port 8000 --reload
```

---

## 🧪 Testing

```bash
# Node test suite (56 tests)
npm --prefix backend test

# Python test suite (61 tests)
pytest -q
```
