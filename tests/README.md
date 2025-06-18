# Voice Assistant Component Testing

This directory contains **real component tests** for the voice assistant project. All tests use actual APIs and hardware - **NO MOCKS**.

## 🧪 Available Tests

### 1. **Gemini Knowledge Base** (`kb`)
- **Location**: `tests/test_kb/test_gemini_kb.py`
- **Tests**: Document ingestion, analysis, retrieval, embeddings, reasoning
- **Requirements**: Gemini API key, internet connection
- **Models Used**: Gemini 2.5 Flash, Gemini 2.5 Pro, text-embedding-004

### 2. **OpenAI Audio** (`audio`)
- **Location**: `tests/test_audio/test_openai_audio.py`
- **Tests**: Text-to-speech, transcription, voice options, performance
- **Requirements**: OpenAI API key, internet connection
- **Models Used**: gpt-4o-transcribe, gpt-4o-mini-tts, tts-1-hd, whisper-1

### 3. **Hardware Audio** (`hardware`)
- **Location**: `tests/test_audio/test_mic_speaker.py`
- **Tests**: Microphone input, speaker output, latency, noise levels
- **Requirements**: Working microphone and speakers
- **Dependencies**: sounddevice, numpy, pygame

### 4. **Wake Word Detection** (`wakeword`)
- **Location**: `tests/test_audio/test_wake_word.py`
- **Tests**: Porcupine wake words, OpenAI real-time detection, voice activity
- **Requirements**: Microphone, API keys (optional Porcupine key)
- **Dependencies**: sounddevice, numpy, openai, pvporcupine (optional)

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Individual Tests
```bash
# Test Gemini knowledge base functionality
python tests/run_tests.py kb

# Test OpenAI audio models
python tests/run_tests.py audio

# Test microphone and speaker hardware
python tests/run_tests.py hardware

# Test wake word detection engines
python tests/run_tests.py wakeword

# Run all tests sequentially
python tests/run_tests.py all
```

### Direct Test Execution
```bash
# Run tests directly (alternative method)
python tests/test_kb/test_gemini_kb.py
python tests/test_audio/test_openai_audio.py
python tests/test_audio/test_mic_speaker.py
python tests/test_audio/test_wake_word.py
```

## 📋 Test Requirements

### API Keys Required
- **OPENAI_API_KEY** - Set in `config/api_keys.env`
- **GEMINI_API_KEY** - Set in `config/api_keys.env`
- **PORCUPINE_ACCESS_KEY** - Optional, for professional wake word detection
  - Get free key at [Picovoice Console](https://console.picovoice.ai/)

### System Requirements
- **Python 3.8+**
- **Working internet connection** (for API tests)
- **Microphone and speakers** (for hardware tests)
- **Audio permissions** (microphone access)

### Dependencies
```bash
# Core AI APIs
google-generativeai>=0.3.0
openai>=1.0.0

# Audio processing
sounddevice>=0.4.6
numpy>=1.24.0
pygame>=2.5.0

# Wake word detection (optional)
pvporcupine  # Professional wake word engine

# Utilities
python-dotenv>=1.0.0
```

## 🎯 What Each Test Does

### Gemini KB Test
- ✅ Creates test documents with project information
- ✅ Tests Gemini 2.5 Flash for document analysis
- ✅ Tests Gemini 2.5 Pro for complex reasoning
- ✅ Generates embeddings for semantic search
- ✅ Tests conversation memory and context
- 🧹 Cleans up test files automatically

### OpenAI Audio Test
- ✅ Generates speech with different voices (alloy, echo, fable, onyx, nova)
- ✅ Tests steerable TTS with emotional styles
- ✅ Tests transcription accuracy with multiple models
- ✅ Measures real-time performance for voice assistant use
- ✅ Creates audio files for manual playback testing
- 💾 Preserves generated files in `tests/test_audio/generated/`

### Hardware Test
- ✅ Lists all available audio devices
- ✅ Records microphone input and analyzes levels
- ✅ Tests speaker output with various frequencies
- ✅ Measures audio latency for real-time use
- ✅ Tests background noise levels
- ✅ Tests simultaneous recording/playback
- 💾 Saves test recordings in `tests/test_audio/hardware/`

### Wake Word Test
- ✅ Tests Porcupine professional wake word engine (if configured)
- ✅ Tests OpenAI real-time wake phrase detection
- ✅ Tests simple voice activity detection
- ✅ Measures detection accuracy and performance
- ✅ Supports multiple wake phrases ("hey jane", "computer", etc.)
- 💾 Saves detection recordings in `tests/test_audio/wake_word/`

## 📊 Test Output

Each test provides:
- **Real-time progress indicators**
- **Detailed performance metrics**
- **Quality assessments**
- **Pass/fail status for each component**
- **Generated files for manual verification**
- **Specific recommendations for issues**

## 🔧 Troubleshooting

### API Test Issues
```bash
# Check API keys
cat config/api_keys.env

# Test basic connectivity
python test_apis.py
```

### Audio Hardware Issues
```bash
# Check audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone permissions (macOS)
# System Preferences > Security & Privacy > Microphone
```

### Dependency Issues
```bash
# Install missing packages
pip install sounddevice numpy pygame

# macOS audio issues
brew install portaudio
pip install pyaudio
```

## 📁 Generated Test Files

Tests create files for manual verification:

### Audio Files
- `tests/test_audio/generated/*.mp3` - OpenAI TTS outputs
- `tests/test_audio/hardware/*.wav` - Hardware test recordings

### Knowledge Base Files
- `kb/test_documents/*.txt` - Temporary test documents (auto-cleaned)

### Manual Testing
```bash
# Play generated audio files (macOS)
cd tests/test_audio/generated
open *.mp3

# Play hardware test files
cd tests/test_audio/hardware  
open *.wav
```

## 🎯 Expected Performance

### API Response Times
- **Gemini 2.5 Flash**: <3 seconds for chat
- **Gemini 2.5 Pro**: <10 seconds for complex reasoning
- **OpenAI TTS**: <2 seconds for short phrases
- **OpenAI Transcription**: <5 seconds for 30s audio

### Hardware Metrics
- **Audio Latency**: <100ms for good performance
- **Noise Floor**: <-30dB for quiet environment
- **Input Levels**: 0.01-0.9 range for good quality

### Wake Word Performance
- **Porcupine Detection**: <200ms latency (if configured)
- **OpenAI Real-time**: 3-second segments with <2s processing
- **Voice Activity**: Real-time threshold detection

## 🚨 Important Notes

### NO MOCK DATA
- All tests use **real API endpoints**
- All tests use **actual hardware devices**
- Tests may consume API credits
- Tests require working internet connection

### API Usage
- Knowledge base tests: ~50-100 tokens per run
- Audio tests: ~5-10 TTS generations per run
- Transcription tests: ~3-5 transcriptions per run

### Privacy
- Test audio recordings are stored locally only
- No sensitive data is sent to APIs
- Generated files can be deleted after testing

## 📈 Next Steps

After successful component testing:
1. **Proceed to Phase 1**: Audio I/O Foundation implementation
2. **Use test results**: Optimize configuration based on performance metrics
3. **Reference generated files**: Use as examples for voice assistant responses

---

**🎉 Ready to test?** Run `python tests/run_tests.py all` to verify all components! 