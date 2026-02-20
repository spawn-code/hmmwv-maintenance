# 🔧 HMMWV Technical Assistant

An AI-powered maintenance and repair guide for High Mobility Multipurpose Wheeled Vehicles (HMMWV / Humvee). Upload your official Technical Manuals (TMs) as PDFs and ask the assistant anything — from step-by-step repair procedures and torque specs to symptom-based diagnostics and PMCS checklists.

Runs entirely on your machine. No cloud required. Supports local Ollama models as well as external providers like OpenAI and Anthropic Claude.

---

## Features

- **RAG-powered answers** — Responses are grounded in your uploaded TM PDFs, not general AI knowledge. Every answer cites the source document and page number.
- **Multi-provider AI** — Choose between:
  - **Ollama (Local)** — fully air-gapped, no API key needed
  - **OpenAI-Compatible** — works with OpenAI, Together AI, Groq, vLLM, LM Studio, and any `/v1/chat/completions` endpoint
  - **Anthropic (Claude)** — Claude API via your Anthropic key
- **Persistent settings** — Your provider, model, URLs, and API keys are saved to disk and restored automatically on next launch. Defaults to Ollama with `gpt-oss:latest`.
- **Three assistant modes:**
  - 💬 **Chat** — General maintenance and repair queries
  - 🔍 **Diagnose** — Describe symptoms and get a structured troubleshooting guide
  - 📋 **PMCS Walkthrough** — Guided Before/During/After-Ops checklists
- **Vehicle & category filters** — Narrow responses to a specific HMMWV variant (M998, M1114, M1151, etc.) and maintenance category (Engine, Brakes, CTIS, etc.)
- **Source references with page images** — Each response shows the exact PDF pages used, rendered as images inline
- **Print-friendly output** — Print any response to a formatted page with embedded diagram images
- **Quick action buttons** — One-click prompts for the most common maintenance tasks
- **PDF management** — Upload TMs directly from the browser; the app tracks which files have been processed and only re-ingests changed files

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 or newer |
| pip | latest |
| Ollama *(if using local AI)* | latest — [ollama.com](https://ollama.com) |

> **No GPU required** for TF-IDF search. A GPU will speed up Ollama inference but is optional.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/spawn-code/hmmwv-maintenance.git
cd hmmwv-maintenance
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Ollama only) Install Ollama and pull a model

Skip this step if you plan to use OpenAI or Anthropic.

```bash
# Install Ollama from https://ollama.com, then:
ollama serve                    # start the Ollama server
ollama pull gpt-oss:latest      # pull the default model
```

Any model available in Ollama will work. The app auto-discovers installed models.

---

## Running the App

### Quick start (recommended)

```bash
./run.sh
```

`run.sh` creates the virtual environment if needed, installs dependencies, and launches Streamlit at `http://localhost:8501`.

### Manual start

```bash
streamlit run app.py
```

---

## First-Time Setup (in the browser)

Once the app is running, open `http://localhost:8501` and follow these three steps:

### Step 1 — Configure your AI provider

In the **sidebar**, select your provider under **AI PROVIDER**:

**Ollama (Local)**
- Enter the Ollama URL (default: `http://localhost:11434`)
- Select your model from the dropdown (or type the model name if offline)

**OpenAI-Compatible**
- Enter the API Base URL (e.g. `https://api.openai.com/v1`)
- Enter the model name (e.g. `gpt-4o`)
- Enter your API key (leave blank for local/no-auth endpoints)

**Anthropic (Claude)**
- Enter your Anthropic API key (`sk-ant-...`)
- Enter the model name (e.g. `claude-opus-4-6`)

Settings are saved automatically and restored on next launch.

### Step 2 — Upload your Technical Manuals

In the sidebar under **KNOWLEDGE BASE**, either:

- Click **Upload TM PDFs** and select your files, or
- Copy PDF files directly into the `knowledge_base/` folder

Then click **Process PDFs**. The app will extract text and images from every page. This only runs once per file — changed files are automatically re-detected via MD5 hash.

Recommended TMs:
- `TM 9-2320-280-10` — Operator's Manual
- `TM 9-2320-280-20` — Unit Maintenance Manual
- `TM 9-2320-280-20P` — Parts Manual

### Step 3 — Ask away

Type your question in the chat box or click a **Quick Action** button. Use the **Vehicle** and **Maintenance Category** dropdowns to focus responses on a specific variant or system.

---

## Project Structure

```
hmmwv-maintenance/
├── app.py                   # Main application (self-contained)
├── requirements.txt         # Python dependencies
├── run.sh                   # One-command launcher
├── knowledge_base/          # Place your TM PDFs here
├── extracted_images/        # Auto-generated page images
├── chroma_db/               # TF-IDF vector store (auto-generated)
└── .hmmwv_settings.json     # Persisted user settings (auto-generated)
```

---

## Configuration Reference

All settings are saved automatically via the UI. For reference, `.hmmwv_settings.json` stores:

| Key | Default | Description |
|---|---|---|
| `provider` | `Ollama (Local)` | Active AI provider |
| `ollama_url` | `http://localhost:11434` | Ollama server URL |
| `ollama_model` | `gpt-oss:latest` | Ollama model name |
| `openai_url` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `openai_model` | `gpt-4o` | OpenAI-compatible model name |
| `openai_api_key` | *(empty)* | OpenAI-compatible API key |
| `anthropic_api_key` | *(empty)* | Anthropic API key |
| `anthropic_model` | `claude-opus-4-6` | Anthropic model name |

> **Note:** API keys are stored in plain text in `.hmmwv_settings.json`. Ensure this file is excluded from version control (it is already in `.gitignore`).

---

## Supported HMMWV Variants

| Designation | Description |
|---|---|
| M998 | Cargo/Troop Carrier |
| M1025 | Armament Carrier |
| M1035 | Soft Top Ambulance |
| M1043 | Up-Armored Armament Carrier |
| M1044 | Up-Armored Armament Carrier w/ Winch |
| M1097 | Heavy HMMWV |
| M1113 | Expanded Capacity Vehicle (ECV) |
| M1114 | Up-Armored HMMWV |
| M1151 | Enhanced Armament Carrier |
| M1152 | Enhanced Cargo/Troop Carrier |
| M1165 | Up-Armored |
| M1167 | TOW Carrier |

---

## Troubleshooting

**Ollama is offline / cannot connect**
- Ensure Ollama is running: `ollama serve`
- Verify the URL in the sidebar matches your Ollama instance
- Check firewall rules if running Ollama on a remote host

**No models appear in the dropdown**
- Pull at least one model: `ollama pull gpt-oss:latest`
- The model name field accepts freeform input when offline

**PDFs produce no text**
- Some scanned PDFs contain only images. Use an OCR tool (e.g. Adobe Acrobat, `ocrmypdf`) to add a text layer before uploading.

**Responses are slow**
- For Ollama: larger models are slower. Try a smaller quantized model.
- For external providers: response time depends on the provider's API latency.

**Reprocess a changed PDF**
- Use **Advanced → Reprocess All** in the sidebar to force re-ingestion of all files.

---

## License

See [LICENSE](LICENSE) for details.
