# Voice-Activated Personal Assistant - Python Implementation Plan

**Based on**: `overall_plan.md` PRD v1.1  
**Target**: Modular Python voice assistant with Gemini KB + OpenAI Audio  
**Timeline**: 6 development phases

---

## Phase 1: Project Setup & Dependencies

### 1.1 Project Structure
```
Jane/
├── src/
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── trigger.py          # Wake word detection
│   │   ├── recorder.py         # Audio capture
│   │   ├── speaker.py          # Audio playback
│   │   └── transcriber.py      # Speech-to-text
│   ├── kb/
│   │   ├── __init__.py
│   │   ├── manager.py          # KB operations
│   │   ├── gemini_client.py    # Gemini API integration
│   │   └── storage.py          # File/embedding management
│   ├── response/
│   │   ├── __init__.py
│   │   ├── generator.py        # OpenAI TTS integration
│   │   └── responder.py        # Response orchestration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── coordinator.py      # Main orchestration
│   │   ├── config.py           # Configuration management
│   │   └── logger.py           # Interaction logging
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py      # Audio format helpers
│       └── file_utils.py       # File operations
├── kb/                         # Knowledge base storage
│   ├── qa_log.jsonl           # Interaction log
│   ├── user/                  # User-specific data
│   └── embeddings/            # Vector embeddings (future)
├── config/
│   ├── settings.yaml          # App configuration
│   └── api_keys.env           # API credentials
├── tests/
│   ├── test_audio/
│   ├── test_kb/
│   └── test_integration/
├── requirements.txt
├── setup.py
└── main.py                    # Application entry point
```

### 1.2 Dependencies Setup
```bash
# Core audio processing
pyaudio>=0.2.11
sounddevice>=0.4.6
pvporcupine>=3.0.0
pygame>=2.5.0

# AI/ML APIs
google-generativeai>=0.3.0
openai>=1.0.0
chromadb>=0.4.0

# Async and utilities
asyncio
aiofiles>=23.0.0
pydantic>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0

# Development/testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
isort>=5.12.0
```

---

## Phase 2: Audio I/O Foundation

### 2.1 Wake Word Detection (`src/audio/trigger.py`)
```python
class WakeWordDetector:
    def __init__(self, keyword="hey_google"):
        self.porcupine = pvporcupine.create(keywords=[keyword])
        self.audio_stream = None
        
    async def listen_for_wake_word(self) -> bool:
        """Continuously listen for wake phrase"""
        # Implementation: pyaudio stream + porcupine detection
        
    def cleanup(self):
        """Clean up audio resources"""
```

**Tasks:**
- [ ] Implement Porcupine wake word detection
- [ ] Add audio device enumeration and selection
- [ ] Handle microphone permissions and errors
- [ ] Test wake word accuracy and latency (<500ms requirement)

### 2.2 Audio Recording (`src/audio/recorder.py`)
```python
class AudioRecorder:
    def __init__(self, sample_rate=16000, max_duration=30):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
    async def record_query(self) -> bytes:
        """Record audio until silence or timeout"""
        # Implementation: VAD + timeout recording
        
    def save_audio(self, audio_data: bytes, filepath: str):
        """Save recorded audio to file"""
```

**Tasks:**
- [ ] Implement voice activity detection (VAD)
- [ ] Add silence detection for auto-stop
- [ ] Handle audio format conversion (WAV/PCM)
- [ ] Test recording quality and duration limits

### 2.3 Audio Playback (`src/audio/speaker.py`)
```python
class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        
    async def play_response(self, audio_data: bytes):
        """Play TTS audio response"""
        # Implementation: pygame/pyaudio playback
        
    def stop_playback(self):
        """Stop current audio playback"""
```

**Tasks:**
- [ ] Implement audio playback with pygame
- [ ] Add volume control and audio device selection
- [ ] Handle audio format compatibility
- [ ] Test playback latency and quality

---

## Phase 3: Speech Processing Integration

### 3.1 Transcription Service (`src/audio/transcriber.py`)
```python
class TranscriptionService:
    def __init__(self, use_openai=True):
        self.openai_client = OpenAI()
        self.gemini_client = genai.GenerativeModel('gemini-pro')
        
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Convert audio to text using OpenAI/Gemini"""
        # Priority: gpt-4o-transcribe > gpt-4o-mini-transcribe > Whisper
        
    async def transcribe_with_fallback(self, audio_data: bytes) -> str:
        """Transcribe with multiple model fallback"""
```

**Tasks:**
- [ ] Integrate OpenAI gpt-4o-transcribe model
- [ ] Add Gemini transcription as fallback
- [ ] Implement error handling and retries
- [ ] Test transcription accuracy (<5s requirement)
- [ ] Add language detection and multi-language support

### 3.2 Response Generation (`src/response/generator.py`)
```python
class ResponseGenerator:
    def __init__(self):
        self.openai_client = OpenAI()
        
    async def generate_speech(self, text: str, instructions: str = "") -> bytes:
        """Generate TTS audio using OpenAI gpt-4o-mini-tts"""
        # Implementation: steerable TTS with voice instructions
        
    def format_voice_instructions(self, context: dict) -> str:
        """Create voice instructions based on context"""
        # Examples: "speak like a helpful assistant", "use empathetic tone"
```

**Tasks:**
- [ ] Integrate OpenAI gpt-4o-mini-tts model
- [ ] Implement voice instruction formatting
- [ ] Add voice personality configuration
- [ ] Test audio generation quality and latency
- [ ] Handle different response types (informational, conversational)

---

## Phase 4: Knowledge Base System

### 4.1 Gemini KB Client (`src/kb/gemini_client.py`)
```python
class GeminiKBClient:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = 'gemini-embedding-exp-03-07'
        
    async def extract_context(self, query: str, kb_content: str) -> dict:
        """Extract relevant context using Gemini"""
        # Returns: {context, summary, tags, file_suggestions}
        
    async def suggest_file_structure(self, content: str) -> dict:
        """Suggest file organization using Gemini"""
        # Returns: {file_path, content_type, tags}
        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for vector search"""
```

**Tasks:**
- [ ] Implement Gemini Pro API integration
- [ ] Add context extraction with structured output
- [ ] Implement file path suggestion logic
- [ ] Add embedding generation for vector search
- [ ] Test context relevance and accuracy

### 4.2 Storage Manager (`src/kb/storage.py`)
```python
class KBStorageManager:
    def __init__(self, kb_root="./kb"):
        self.kb_root = Path(kb_root)
        self.qa_log_path = self.kb_root / "qa_log.jsonl"
        
    async def log_interaction(self, interaction: dict):
        """Append interaction to qa_log.jsonl"""
        
    async def save_structured_content(self, content: str, file_path: str):
        """Save content to suggested markdown/JSON files"""
        
    def search_kb_files(self, query: str) -> List[str]:
        """Search existing KB files for relevant content"""
        
    async def load_relevant_context(self, query: str) -> str:
        """Load and combine relevant KB content"""
```

**Tasks:**
- [ ] Implement JSONL logging system
- [ ] Add file-based storage for markdown/JSON
- [ ] Create file search and indexing
- [ ] Add content retrieval and combination
- [ ] Test storage performance and organization

---

## Phase 5: Core Orchestration

### 5.1 Main Coordinator (`src/core/coordinator.py`)
```python
class VoiceAssistantCoordinator:
    def __init__(self):
        self.wake_detector = WakeWordDetector()
        self.recorder = AudioRecorder()
        self.transcriber = TranscriptionService()
        self.kb_manager = KBManager()
        self.response_gen = ResponseGenerator()
        self.speaker = AudioPlayer()
        
    async def run_assistant_loop(self):
        """Main event loop for voice assistant"""
        while True:
            await self.wake_detector.listen_for_wake_word()
            await self.handle_voice_query()
            
    async def handle_voice_query(self):
        """Complete voice interaction pipeline"""
        # 1. Record audio
        # 2. Transcribe to text
        # 3. Query KB for context
        # 4. Generate response
        # 5. Play audio response
        # 6. Log interaction
```

**Tasks:**
- [ ] Implement complete voice interaction pipeline
- [ ] Add error handling and recovery
- [ ] Implement timeout management
- [ ] Add performance monitoring
- [ ] Test end-to-end latency (<8s requirement)

### 5.2 Configuration Management (`src/core/config.py`)
```python
@dataclass
class VoiceAssistantConfig:
    # Audio settings
    wake_word: str = "hey_google"
    sample_rate: int = 16000
    max_recording_duration: int = 30
    
    # API settings
    openai_api_key: str = ""
    gemini_api_key: str = ""
    
    # KB settings
    kb_root_path: str = "./kb"
    max_context_length: int = 4000
    
    # Performance settings
    max_response_latency: float = 8.0
    transcription_timeout: float = 5.0
```

**Tasks:**
- [ ] Create configuration data classes
- [ ] Add YAML configuration file support
- [ ] Implement environment variable loading
- [ ] Add configuration validation
- [ ] Create configuration update mechanisms

---

## Phase 6: Testing & Integration

### 6.1 Unit Tests
```python
# tests/test_audio/test_transcriber.py
class TestTranscriptionService:
    async def test_transcribe_audio_openai(self):
        """Test OpenAI transcription accuracy"""
        
    async def test_transcribe_with_fallback(self):
        """Test fallback to alternative models"""

# tests/test_kb/test_gemini_client.py
class TestGeminiKBClient:
    async def test_extract_context(self):
        """Test context extraction quality"""
        
    async def test_suggest_file_structure(self):
        """Test file organization suggestions"""
```

### 6.2 Integration Tests
```python
# tests/test_integration/test_end_to_end.py
class TestVoiceAssistantE2E:
    async def test_complete_voice_interaction(self):
        """Test full pipeline from wake word to response"""
        
    async def test_performance_requirements(self):
        """Test latency and accuracy requirements"""
```

### 6.3 Performance Testing
**Target Metrics:**
- Wake word detection: <500ms
- Audio transcription: <5s
- KB context retrieval: <2s
- Response generation: <3s
- Total end-to-end: <8s

---

## Implementation Checklist

### Sprint 1: Foundation (Week 1-2)
- [ ] Set up project structure and dependencies
- [ ] Implement wake word detection
- [ ] Create basic audio recording/playback
- [ ] Add configuration management

### Sprint 2: Speech Processing (Week 3-4)
- [ ] Integrate OpenAI transcription models
- [ ] Implement TTS response generation
- [ ] Add error handling and fallbacks
- [ ] Test audio quality and latency

### Sprint 3: Knowledge Base (Week 5-6)
- [ ] Implement Gemini KB client
- [ ] Create storage management system
- [ ] Add file organization and search
- [ ] Test context extraction accuracy

### Sprint 4: Integration (Week 7-8)
- [ ] Build main coordinator system
- [ ] Implement complete interaction pipeline
- [ ] Add logging and monitoring
- [ ] Conduct end-to-end testing

### Sprint 5: Optimization (Week 9-10)
- [ ] Performance tuning and optimization
- [ ] Error handling improvements
- [ ] User experience enhancements
- [ ] Documentation completion

### Sprint 6: Cloud Migration Prep (Week 11-12)
- [ ] Design cloud storage architecture
- [ ] Implement vector embedding system
- [ ] Create migration tools
- [ ] Test cloud integration prototype

---

## Development Commands

```bash
# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run development server
python main.py

# Run tests
pytest tests/ -v

# Code formatting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Performance testing
python -m pytest tests/test_integration/test_performance.py -v
```

---

## Migration Path to Cloud

### Local → Cloud Storage
1. **File Storage**: `./kb/` → `gs://voice-assistant-kb/`
2. **Embeddings**: Local Chroma → Cloud Chroma/FAISS
3. **API Keys**: Local env → Google Secret Manager
4. **Monitoring**: Local logs → Google Cloud Logging

### Architectural Changes Required
- Add GCS client integration
- Implement cloud authentication
- Update storage manager for cloud operations
- Add distributed embedding search

This plan provides a clear roadmap for implementing the voice assistant with all specified requirements while maintaining modularity for future cloud migration. 