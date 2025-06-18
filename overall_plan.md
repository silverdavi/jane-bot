

---

# **Product Requirements Document (PRD)**

## Voice-Activated Personal Assistant with Gemini-Enhanced KB

**Owner**: David Silver
**Date**: June 17, 2025
**Version**: 1.1

---

## 1. **Objective**

Develop a modular Python-based voice assistant that:

* Activates on a specific wake phrase,
* Converts voice input to text,
* Retrieves and updates a knowledge base (KB) using Gemini,
* Generates a voice response using OpenAI's audio API,
* Logs all interactions for future use.

The system is **initially local-first**, with a clear migration path to **cloud-based storage and vector indexing**.

---

## 2. **System Architecture Overview**

```plaintext
 [Microphone Input] 
       ↓
[Wake Word Detection]
       ↓
[Audio Capture → Transcription]
       ↓
[Gemini KB Query → Context Extraction]
       ↓
[OpenAI Audio API → Speech Output]
       ↓
[Response Played + All Data Logged]
```

---

## 3. **System Components**

### 3.1. **Wake Word + Audio I/O Subsystem**

* **Function**: Detects wake phrase ("Hi Chat Gee Pee Tee"), captures audio.
* **Modules**:

  * `trigger.py`: Uses local wake-word engine (Porcupine or equivalent).
  * `recorder.py`: Captures a voice query (\~30 seconds).
  * `speaker.py`: Plays audio response.
* **Tools**: `sounddevice`, `pyaudio`, `porcupine`, `playsound`

---

### 3.2. **Speech-to-Speech Orchestration Layer**

* **Function**: Orchestrates input transcription, KB context retrieval, and response generation.
* **Modules**:

  * `transcriber.py`: Whisper or Gemini-based transcription.
  * `responder.py`: Builds prompt and sends to OpenAI audio model.
  * `coordinator.py`: Ties together I/O, KB, and response.

---

### 3.3. **Knowledge Base Management (LLM-Centric)**

* **Function**: Use Gemini to extract, search, and store relevant Q\&A data.
* **Current Implementation**:

  * Local `qa_log.jsonl` + structured markdown (`user/job.md`, `user/data/info.json`)
  * Gemini API for context extraction and smart storage suggestions
* **Future Cloud Migration**:

  * Markdown/JSON → Google Cloud Storage (`gs://voice-assistant-kb/users/...`)
  * Embeddings → Gemini Text Embedding API → FAISS/Chroma (hosted or GCS-mount)

---

## 4. **Data Storage Strategy**

| Layer       | Format             | Location                   | Notes                             |
| ----------- | ------------------ | -------------------------- | --------------------------------- |
| Q\&A Log    | JSONL              | `./kb/qa_log.jsonl`        | Append-only log                   |
| Structured  | Markdown/JSON      | `./kb/user/*.md`           | Per-topic file separation         |
| Embeddings  | Chroma DB          | `./kb/embeddings/`         | Optional – stored locally for now |
| Cloud-ready | GCS + Chroma/FAISS | `gs://voice-assistant-kb/` | Future plan, same structure       |

---

## 5. **Functional Requirements**

| Feature             | Requirement                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| Wake-word listener  | Must respond to “Hi Chat Gee Pee Tee” with latency < 500ms               |
| Audio transcription | Must complete transcription of <30s audio in <5s                         |
| KB retrieval        | Gemini must return structured context, summary, and tags                 |
| Response generation | Audio response must be playable with <8s end-to-end latency              |
| Logging             | Full interaction history must be stored locally in structured format     |
| File organization   | Gemini should decide file paths like `user/job.md`, `user/data/job.json` |
| Local operation     | All systems must run offline (except Gemini/OpenAI API calls)            |

---

## 6. **Non-Functional Requirements**

* **Latency**: End-to-end voice query to response playback under 8 seconds.
* **Scalability**: System should support upgrade to cloud storage and remote KB without structural change.
* **Portability**: Works on Linux mini PC with Python 3.10+, USB mic/speaker.
* **Resilience**: Recover from wake word failure, network drop, or API timeout.

---

## 7. **Stretch Goals**

* Conversational memory across sessions (threaded Q\&A)
* GUI or CLI dashboard for reviewing past logs
* Real-time barge-in interrupt
* Gemini-based summarization of entire daily sessions

---

## 8. **Tech Stack**

| Layer             | Tech                           |
| ----------------- | ------------------------------ |
| Audio I/O         | PyAudio, Porcupine, Playsound  |
| Transcription     | Whisper / Gemini ASR           |
| Speech Generation | OpenAI Audio API               |
| KB Logic          | Gemini Chat API                |
| Embedding Index   | Gemini Embeddings + Chroma     |
| Storage           | Local filesystem (future: GCS) |
| Orchestration     | Python + asyncio               |

---

## 9. **Milestones**

| Milestone                        | 
| -------------------------------- | 
| Wake-word detection + I/O tested | 
| Speech-to-speech loop built      |
| Gemini KB retrieval integrated   | 
| Markdown + JSON file handling    | 
| Local embedding + logging        | 
| Cloud bucket + Chroma prototype  | 

---

