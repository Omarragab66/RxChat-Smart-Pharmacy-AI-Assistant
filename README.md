<div align="center">

# 🩺 RxChat — Smart Pharmacy AI Assistant

**RAG-Powered Pharmaceutical Assistant for Egyptian Arabic Speakers**

[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=flat-square&logo=node.js&logoColor=white)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.8--3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3-F54E00?style=flat-square)](https://groq.com/)
[![AraBERT](https://img.shields.io/badge/NLP-AraBERT%20v2-4A90D9?style=flat-square)](https://huggingface.co/aubmindlab/bert-base-arabertv2)
[![License](https://img.shields.io/badge/License-Educational-lightgrey?style=flat-square)](./LICENSE)

*Graduation Project 2024/2025 — AI & Computer Science*

</div>

---

## 📖 Overview

RxChat is a **Retrieval-Augmented Generation (RAG)** pharmaceutical assistant that answers drug-related questions in **Egyptian Arabic dialect** using verified data from OpenFDA and RxNorm. Unlike a generic chatbot, it combines four deep learning modules to provide personalized, safe, and hallucination-free medical information.

### Why RxChat?

| Generic RAG Chatbot | RxChat |
|---|---|
| Answers from general knowledge | Answers strictly from OpenFDA + RxNorm data |
| May hallucinate | Says *"I don't know"* when unsure |
| English only | Fine-tuned for Egyptian Arabic (عامية) |
| Standard FDA dosage | Personalized dosage by age, weight & conditions |
| No safety check | AraBERT safety classifier (Low / Medium / High risk) |

---

## ✨ Features

- 💬 **Conversational AI** — Medical guidance in Egyptian Arabic dialect and English
- 🛡️ **AI Safety Classifier** — AraBERT-based model classifies queries as Low / Medium / High risk
- 💊 **Personalized Dosage Calculator** — DeepSeek-R1 chain-of-thought reasoning adjusted per patient profile (age, weight, kidney/liver function)
- 🔍 **Clinical Arabic RAG** — ChromaDB vector retrieval from cleaned OpenFDA drug embeddings
- 🎙️ **Voice TTS** — Multilingual text-to-speech via ElevenLabs (`eleven_multilingual_v2`)
- 🌐 **Modern UI** — Glassmorphism styling, real-time medical logging dashboard
- 🔒 **No Hallucination** — LLM is strictly prompted to answer only from retrieved context

---

## 🏗️ System Architecture

```
User Input (Arabic / English)
         │
         ▼
   FastAPI + Node.js Backend
         │
    ┌────┴────┐
    │         │
  Text      Image
  Input    Upload
    │         │
    │    Vision LLM
    │    (Multi-model voting)
    │         │
    └────┬────┘
         │
         ▼
  ┌──────────────────────────────────┐
  │          RAG Pipeline            │
  │                                  │
  │  1. AraBERT NLP Layer            │  ← Person 2 (Safety + Intent)
  │     dialect normalization,       │
  │     intent detection, NER        │
  │                                  │
  │  2. Embedder → ChromaDB          │  ← Person 4 (OpenFDA embeddings)
  │     semantic retrieval           │
  │                                  │
  │  3. Dosage Calculator            │  ← Person 3 (DeepSeek-R1 CoT)
  │     patient-adjusted dose        │
  │                                  │
  │  4. Prompt Builder               │  ← Person 4 (LangChain)
  │     context + safety + dosage    │
  │                                  │
  │  5. Groq LLM (Llama 3.3)         │
  │     final grounded response      │
  └──────────────────────────────────┘
         │
         ▼
  Response + Drug Image → Frontend
```

---

## 🧠 Deep Learning Modules

### Module 1 — Arabic Safety Classifier (Person 2)
Fine-tuned **AraBERT v2** (`aubmindlab/bert-base-arabertv2`) for medical query safety classification.

- **Architecture**: BERT (12 layers, 768 hidden, 12 attention heads, 64k vocab)
- **Task**: Multi-class sequence classification → `Low` / `Medium` / `High` risk
- **Training data**: Custom Egyptian Arabic medical dataset (`p2_safety_dataset.csv`, `arabert_new.csv`)
- **Serving**: Flask API (`person2_api.py`) with `flask-cors`, PyTorch inference
- **Model artifacts**: `rxchat_safety_model_final_fixed/` (tokenizer + weights + label mapping)

```python
# Label mapping
{ "Low": 0, "Medium": 1, "High": 2 }
```

---

### Module 2 — Personalized Dosage Calculator (Person 3)
Chain-of-thought reasoning with **DeepSeek-R1 via Groq API**, backed by a rule-based scikit-learn safety classifier.

- **Architecture**: Few-shot CoT prompting → DeepSeek-R1 + RandomForest pre-filter
- **Training data**: 169 few-shot examples (`Deep_seek_data.json`)
- **Pre-trained artifact**: `safety_classifier.pkl` (RandomForest)
- **Serving**: FastAPI on port `8003` (`dosage_calculator.py`)
- **Patient profile inputs**: age, weight (kg), gender, chronic conditions, kidney function, liver function

**Request / Response example:**
```json
POST /dosage
{
  "drug_name": "Paracetamol",
  "patient_profile": {
    "age": 35, "weight_kg": 70, "gender": "أنثى",
    "kidney_function": "normal", "liver_function": "normal"
  }
}
→ { "recommended_dose": "500-1000 مجم كل 4-6 ساعات",
    "risk_category": "LOW", "safety_flag": false, "doctor_referral": false }
```

---

### Module 3 — RAG Core + Orchestration (Person 4)
Full LangChain pipeline, ChromaDB vector store, multi-model voting for image recognition, and all integrations.

- **Data**: Cleaned OpenFDA drug data (`cleaned_drugs_core.csv`, `Core_data.csv`) embedded into ChromaDB
- **Retrieval**: Semantic search via LangChain embedder + ChromaDB retriever
- **Image recognition**: Multi-model voting — Groq Vision + Gemini Vision + Claude Vision (majority 2/3 wins)
- **Memory**: `ConversationBufferMemory` for session context
- **Notebooks**: `Main_notebook.ipynb`, `RxChat_Integration_Test_PERSON2_API_FIXED_final.ipynb`
- **Bridge**: `rxchat_bridge.py` — Python orchestrator spawned by the Node.js server

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | Groq — Llama 3.3 | Text generation (grounded RAG response) |
| Reasoning LLM | DeepSeek-R1 (Groq) | Personalized dosage chain-of-thought |
| Vision | Groq Vision + Gemini + Claude | Drug image multi-model voting |
| NLP | AraBERT v2 (HuggingFace) | Arabic safety classification |
| RAG | LangChain + ChromaDB | Retrieval pipeline + vector store |
| ML | scikit-learn (RandomForest) | Rule-based dosage pre-filter |
| Backend | Node.js 18+ (server.js) | HTTP server, TTS proxy, bridge process |
| Python Bridge | FastAPI + Python 3.8–3.10 | DL module orchestration |
| TTS | ElevenLabs (`eleven_multilingual_v2`) | Multilingual voice output |
| Frontend | HTML5 + CSS3 + Vanilla JS | Glassmorphism chat UI |
| Deployment | Vercel (frontend) + Render (backend) | Production hosting |
| Data | OpenFDA API + RxNorm | Official drug database |

---

## 📂 Project Structure

```
MedChat/
├── index.html                    # Frontend — glassmorphism chat UI
├── script.js                     # Client-side JS — chat, TTS, API calls
├── styles.css                    # CSS — glassmorphism + animations
├── server.js                     # Node.js HTTP server (port 3000)
├── rxchat_bridge.py              # Python orchestrator (spawned by server.js)
├── safety_classifier.pkl         # Pre-trained RandomForest (dosage safety)
├── package.json                  # Node.js config (node >=18, npm start/dev)
├── .env.example                  # Environment variable template
│
├── api/
│   ├── health.js                 # Serverless health-check endpoint
│   └── tts.js                    # Serverless TTS endpoint (Vercel)
│
├── Data/
│   ├── pers_one/
│   │   ├── p1_intent.csv         # Intent classification dataset
│   │   └── p1_normalization.csv  # Dialect normalization dataset
│   │
│   ├── pers_two/
│   │   ├── arabert_new.csv       # AraBERT training data
│   │   ├── p2_safety_dataset.csv # Safety classification dataset
│   │   ├── person2_api.py        # Flask API — safety inference
│   │   ├── rxchat_safety_model_final/        # AraBERT tokenizer (v1)
│   │   └── rxchat_safety_model_final_fixed/  # AraBERT model + weights (v2)
│   │       ├── config.json
│   │       ├── label_mapping.json
│   │       ├── tokenizer.json
│   │       └── vocab.txt
│   │
│   ├── pers_three/
│   │   ├── dosage_calculator.py  # FastAPI dosage service (port 8003)
│   │   ├── Deep_seek_data.json   # 169 few-shot CoT examples
│   │   ├── safety_classifier.pkl # RandomForest pre-filter
│   │   ├── requirements.txt      # Python deps for Person 3
│   │   └── README (3).md         # Person 3 module documentation
│   │
│   └── pers_foure/
│       ├── Core_data.csv         # Raw OpenFDA drug dataset
│       ├── Core_data_Amr.csv     # Supplementary drug data
│       ├── cleaned_drugs_core.csv         # Cleaned drug embeddings source
│       ├── clean_drugs_mistral.py         # Data cleaning script
│       └── rag_retrieval_test_report.csv  # RAG evaluation results
│
├── Notebooks/
│   ├── Main_notebook.ipynb                              # Core DL training
│   ├── RxChat_Integration_Test_PERSON2_API_FIXED_final.ipynb  # Active runtime
│   ├── RxChat_Integration_Test_FIXED.ipynb
│   ├── RxChat_Integration_Test_MINIMAL_ORCHESTRATOR_FIXED.ipynb
│   ├── RxChat_RAG_Kaggle_Cleaned_CoreData.ipynb
│   ├── final.ipynb
│   ├── llamaIndex.ipynb
│   └── old/                      # Earlier experimental notebooks
│
├── RxChat_Project_Plan.pdf       # Project timeline and plan
└── RxChat_ProjectDoc_v2.docx     # Full project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites

- **Node.js** v18 or higher
- **Python** 3.8–3.10 (3.11 works; avoid 3.12+ for transformer compatibility)
- **pip** with PyTorch, Transformers, Groq SDK

### 1. Clone the repository

```bash
git clone https://github.com/mohmadAyman75/MedChat.git
cd MedChat
```

### 2. Install Node.js dependencies

```bash
npm install
```

### 3. Install Python dependencies

```bash
# Core (bridge + orchestrator)
pip install groq langchain chromadb fastapi uvicorn flask flask-cors

# Deep Learning modules
pip install torch transformers scikit-learn pandas numpy joblib pydantic

# Person 3 specific
pip install -r Data/pers_three/requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
PORT=3000
MEDCHAT_DISPLAY_NAME=Medchat

# Groq API (free tier — required)
GROQ_API_KEY=your_groq_api_key_here
RXCHAT_USE_EXTERNAL_LLM=1

# ElevenLabs TTS (optional)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128

# Module toggles (set to 1 to enable when APIs are running)
RXCHAT_USE_PERSON2_API=0
RXCHAT_USE_PERSON3_API=0

# Timeouts (ms)
RXCHAT_GROQ_TIMEOUT=45
RXCHAT_PIPELINE_TIMEOUT=75
```

### 5. (Optional) Start the Python module APIs

Person 2 — Safety Classifier (port not fixed; set in `person2_api.py`):
```bash
cd Data/pers_two
python person2_api.py
```

Person 3 — Dosage Calculator (port 8003):
```bash
cd Data/pers_three
python dosage_calculator.py
```

### 6. Start RxChat

```bash
npm start          # production
npm run dev        # development (auto-restart on changes)
```

Open `http://localhost:3000` in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Server + bridge status check |
| `POST` | `/api/chat` | Main chat endpoint — RAG pipeline |
| `POST` | `/api/tts` | Text-to-speech via ElevenLabs |
| `POST` | `/dosage` | Personalized dosage (Person 3, port 8003) |
| `POST` | `/predict` | Safety classification (Person 2) |

---

## 🔑 Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | Node.js server port |
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `RXCHAT_USE_EXTERNAL_LLM` | `1` | Enable Groq LLM (set `0` to stub) |
| `ELEVENLABS_API_KEY` | — | ElevenLabs TTS key (optional) |
| `ELEVENLABS_VOICE_ID` | `JBFqnCBsd6RMkjVDRZzb` | ElevenLabs voice |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` | TTS model |
| `RXCHAT_USE_PERSON2_API` | `0` | Enable safety classifier API |
| `RXCHAT_USE_PERSON3_API` | `0` | Enable dosage calculator API |
| `RXCHAT_USE_DIRECT_PERSON3_LLM` | `0` | Call dosage LLM directly |
| `RXCHAT_GROQ_TIMEOUT` | `45` | Groq call timeout (seconds) |
| `RXCHAT_PIPELINE_TIMEOUT` | `75` | Full pipeline timeout (seconds) |
| `RXCHAT_ALLOW_NOTEBOOK_DEFAULT_KEYS` | `0` | Allow keys from notebook cells |
| `RXCHAT_WORKER_THREADS` | `4` | Python bridge thread pool size |

---

## 🗓️ Development Timeline

| Week | Person 1 (NLP Data) | Person 2 (AraBERT Safety) | Person 3 (Dosage) | Person 4 (RAG Core) |
|---|---|---|---|---|
| 1 | Data collection | Data collection | Schema + data | ChromaDB setup |
| 2 | Graph construction | Data labeling | Feature engineering | RAG pipeline |
| 3 | NLP training | Preprocessing | CoT examples | Multi-model voting |
| 4 | NLP training | Fine-tuning AraBERT | Fine-tuning DeepSeek | Module integration |
| 5 | Evaluation | Fine-tuning | Evaluation | FastAPI endpoints |
| 6 | API + integration | API + integration | API + integration | Frontend build |
| 7–8 | Support + testing | Support + testing | Support + testing | Deploy + QA |

---

## 👥 Team

| Role | Module |
|---|---|
| NLP Data Engineer | Intent classification datasets, dialect normalization |
| Arabic Safety Engineer (Person 2) | AraBERT fine-tuning, Flask safety API |
| Dosage Engineer (Person 3) | DeepSeek-R1 CoT dosage API, RandomForest classifier |
| RAG + Orchestration Engineer (Person 4) | LangChain pipeline, Node.js server, frontend, deployment |

---

## ⚠️ Disclaimer

RxChat is an experimental AI assistant developed for **educational and research purposes only**. It is **not a substitute** for professional medical advice, diagnosis, or treatment. Always consult a licensed physician or certified pharmacist for medical concerns.

No patient data is stored permanently. Session memory is cleared after each conversation.

---

<div align="center">

Made in Egypt 🇪🇬 — RxChat Graduation Project 2024/2025

</div>
