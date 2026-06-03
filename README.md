# 🎀 Lecture Q&A — RAG-Based Video Lecture Assistant

Ask any question about your video lectures in plain English.  
Get a written answer **and** jump straight to the exact moment in the video where it's taught.

---

## 🌸 The Problem

Every semester I have hours of recorded lectures. When exam time comes, I waste so much time scrubbing through hour-long videos just to find one concept explained for two minutes. There was no way to search _inside_ a video by meaning.

So I built one.

---

## ✨ What It Does

- Type any question about your lectures
- The system finds the **5 most relevant moments** across all your videos
- An AI writes a clean answer telling you exactly **which lecture and timestamp** covers that topic
- Click the embedded video player and it **jumps straight to that second** — no scrubbing

---

## 🔧 How It Works

This project is a **RAG system** — Retrieval Augmented Generation. It works in two phases:

### Phase 1 — Preparation (runs once)

```
Videos → Audio (FFmpeg) → Text Chunks with Timestamps (Whisper) → Meaning Numbers (BGE-M3) → embeddings.joblib
```

1. **Extract audio** — FFmpeg strips the audio from each lecture video into MP3 files
2. **Transcribe and chunk** — OpenAI Whisper listens to each MP3 and converts speech to text, naturally breaking it into segments with start and end timestamps
3. **Create embeddings** — Each text chunk is converted into a list of ~1000 numbers by the BGE-M3 model. These numbers capture the _meaning_ of the text, not just the words. All 1585 chunks are saved to `embeddings.joblib`

### Phase 2 — Querying (runs every time a user asks)

```
Question → Embedding → Cosine Similarity Search → Top 5 Chunks → LLM Answer → Streamlit UI
```

1. The user's question is converted to numbers using the same BGE-M3 model
2. Cosine similarity (scikit-learn) compares the question's numbers to all 1585 chunk numbers
3. The 5 closest chunks are retrieved — these are the most semantically relevant moments
4. Timestamps are pre-formatted in Python (e.g. `61.16s → 1m 1s`) before being sent to the LLM, to prevent arithmetic hallucination
5. Llama 3.2 1B (via Ollama) generates a streaming answer using only those 5 chunks
6. Streamlit displays the answer and renders each video clip starting at the exact timestamp

---

## 🛠️ Tech Stack

| Tool               | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| **Python**         | Core language                                       |
| **FFmpeg**         | Video to audio extraction                           |
| **OpenAI Whisper** | Speech-to-text transcription with timestamps        |
| **Ollama**         | Local AI runtime (runs models on your own machine)  |
| **BGE-M3**         | Embedding model — converts text to meaning vectors  |
| **Llama 3.2 1B**   | Language model — writes the final answer            |
| **scikit-learn**   | Cosine similarity search                            |
| **NumPy**          | Vector math operations                              |
| **pandas**         | Managing chunks as a structured table               |
| **joblib**         | Saving and loading large numerical data efficiently |
| **Streamlit**      | Web interface                                       |

> Everything runs **100% locally**. No API keys. No subscriptions. No data leaves your machine.

---

## 📁 Project Structure

```
RAG/
├── app.py                  # Streamlit web app (main interface)
├── process_video.py        # Step 1: extract audio from videos
├── chunks.py               # Step 2: transcribe audio into chunks
├── read_chunks.py          # Step 3: create embeddings and save
├── process_incoming.py     # CLI version of the query pipeline
├── embeddings.joblib       # Saved embeddings database (all chunks)
├── audios/                 # Extracted MP3 audio files
├── jsons/                  # Chunked transcripts with timestamps
├── Videos/                 # Original lecture video files
└── requirements.txt        # Python dependencies
```

---

## 🚀 Setup and Running

### Prerequisites

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed on your system
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/RAG-Model.git
cd RAG-Model
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Pull the required Ollama models

```bash
ollama pull bge-m3
ollama pull llama3.2:1b
```

### 4. Add your lecture videos

Place your video files inside the `Videos/` folder.  
Name them like: `Lecture 01 intro.mp4`, `Lecture 02 basics.mp4`

### 5. Run the preparation pipeline (once)

```bash
# Step 1 — extract audio from videos
python3 process_video.py

# Step 2 — transcribe and chunk
python3 chunks.py

# Step 3 — create embeddings
python3 read_chunks.py
```

This creates `embeddings.joblib` — your searchable database. You only need to do this once, or again when you add new videos.

### 6. Launch the web app

```bash
# Make sure Ollama is running
ollama serve

# In a new terminal, launch the app
python3 -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 💬 Example Queries

- _"Where is Triveni Cafe discussed?"_
- _"What is residential architecture?"_
- _"Which lecture covers interior design case studies?"_

---

## 🧠 Key Design Decisions

**Why local models?** Privacy and cost. All processing happens on your own machine. Your lecture content never leaves your device.

**Why Whisper for chunking?** Whisper's natural speech segments (with timestamps) are far better chunks than arbitrary character splits, because they follow how speech actually flows.

**Why pre-format timestamps in Python?** Small LLMs (1B parameters) are unreliable at arithmetic. Formatting timestamps in code before passing them to the model eliminates hallucination on this specific task.

**Why joblib over CSV/JSON for embeddings?** Embeddings are large float arrays. Joblib handles them efficiently — faster to save, faster to load, smaller on disk than JSON.

---

## 👤 Built By

**Shabadpreet Singh**  
B.Arch, IIT Roorkee  
Built as part of an application to MDG Space, IIT Roorkee

---

## 📄 License

MIT License — free to use, modify, and share.
