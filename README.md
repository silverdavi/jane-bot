# 🤖 Jane Voice Assistant

**Advanced voice assistant with dual-thread architecture featuring foreground conversation and background ambient transcription.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper%20%7C%20GPT--4o-green.svg)](https://openai.com/)
[![Google](https://img.shields.io/badge/Google-Gemini%202.5-blue.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

### 🎯 **Dual-Thread Architecture**
- **FOREGROUND**: Wake word detection and conversation management
- **BACKGROUND**: Continuous ambient transcription with intelligent summaries
- **NO CONFLICTS**: Separate threads prevent task interference

### 🗣️ **Advanced Voice Capabilities**
- 🎤 **Wake Words**: "hey jane", "jane", "computer"
- 🧠 **AI Models**: OpenAI Whisper (STT) + Gemini 2.5 (reasoning) + GPT-4o (TTS)
- 🔊 **Multi-Voice TTS**: 5 voices with emotional tone adaptation
- 📊 **Voice Activity Detection**: Natural pause recognition

### 📚 **Intelligent Knowledge Management**
- 💾 **Persistent Buffer**: Sliding window transcription storage
- 🔄 **Context Summaries**: 45-second intelligent summaries with overlap
- 📖 **Self-Aware**: Jane knows her own capabilities and can answer questions about herself
- 🔍 **Smart Search**: Enhanced knowledge base search with contextual results

### 🎨 **Rich User Experience**
- 🌈 **Visual Feedback**: Colored state indicators and progress bars
- 🔊 **Audio Feedback**: Different tones for each state transition
- ⏱️ **Real-Time Updates**: Live transcription and processing status
- 🛑 **Smart Controls**: "STOP DEMO" command works in all states

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Microphone and speakers/headphones
- OpenAI API key
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/silverdavi/jane-bot.git
   cd jane-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   cp config/api_keys.env.template config/api_keys.env
   # Edit config/api_keys.env and add your API keys
   ```

5. **Run the demo**
   ```bash
   python demo_voice_assistant.py
   ```

## 🎯 Usage

### Basic Commands
- **Start conversation**: Say "hey jane", "jane", or "computer"
- **Ask questions**: Natural language after wake word detection
- **Stop demo**: Say "stop demo" at any time
- **Get help**: Ask Jane about her capabilities

### Example Conversations
```
👤 "Hey Jane"
🤖 [Wake detected beep]

👤 "What are your wake words?"
🤖 "My wake words are 'hey jane', 'hello jane', 'jane', and 'computer'. You can use any of these to start a conversation with me!"

👤 "Computer"
🤖 [Wake detected beep]

👤 "How does continuous transcription work?"
🤖 "I use background transcription with sliding windows that process 15 segments at a time with 5-segment overlap for context continuity!"
```

## 🧪 Testing

Run comprehensive tests for all components:

```bash
# Run all tests
python tests/run_tests.py all

# Run specific test suites
python tests/run_tests.py kb          # Knowledge base tests
python tests/run_tests.py audio      # Audio processing tests
python tests/run_tests.py hardware   # Hardware audio tests
python tests/run_tests.py wakeword   # Wake word detection tests
```

## 📋 API Testing

Test API connectivity and model availability:

```bash
python test_apis.py
```

## 🏗️ Architecture

### Main Components

- **`demo_voice_assistant.py`**: Main demo application with dual-thread architecture
- **`src/`**: Core voice assistant modules (planned for Phase 2)
- **`tests/`**: Comprehensive test suite with real API integration
- **`config/`**: Configuration files and API key templates
- **`kb/`**: Knowledge base with documentation and user data

### AI Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Wake Word Detection | OpenAI Whisper-1 | Real-time wake phrase recognition |
| Question Transcription | OpenAI Whisper-1 | High-accuracy speech-to-text |
| Question Analysis | Google Gemini 2.5 Flash | Fast question understanding |
| Complex Reasoning | Google Gemini 2.5 Pro | Advanced reasoning tasks |
| Response Polishing | OpenAI GPT-4o-Mini | Natural conversation optimization |
| Speech Synthesis | OpenAI TTS (5 voices) | Emotional text-to-speech |
| Background Summarization | Google Gemini 2.5 Flash | Ambient content analysis |

## 🔧 Configuration

### Audio Settings
- **Sample Rate**: 16kHz
- **Recording Timeout**: 20 seconds maximum
- **Silence Detection**: 2 seconds stops recording
- **Background Summaries**: Every 45 seconds

### Voice Options
- **Nova**: Helpful (default)
- **Alloy**: Friendly
- **Echo**: Professional
- **Fable**: Enthusiastic
- **Onyx**: Apologetic

## 📁 Project Structure

```
jane-bot/
├── 🤖 demo_voice_assistant.py     # Main demo application
├── 📋 test_apis.py                # API connectivity testing
├── 📝 requirements.txt            # Python dependencies
├── ⚙️ config/
│   ├── api_keys.env.template      # API key template
│   └── settings.yaml              # Application settings
├── 🧪 tests/                      # Comprehensive test suite
│   ├── run_tests.py               # Test runner
│   ├── test_audio/                # Audio processing tests
│   └── test_kb/                   # Knowledge base tests
├── 📚 kb/                         # Knowledge base
│   └── user/                      # User data and documentation
├── 📖 API_documentations/         # API research and documentation
└── 🏗️ src/                       # Core modules (future development)
```

## 🛠️ Development

### Running Individual Components

1. **API Testing**
   ```bash
   python test_apis.py
   ```

2. **Knowledge Base Testing**
   ```bash
   python tests/test_kb/test_gemini_kb.py
   ```

3. **Audio Testing**
   ```bash
   python tests/test_audio/test_openai_audio.py
   ```

### Adding New Features

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and make sure all tests pass before submitting a pull request.

### Development Principles
- **No Mocks**: All tests use real APIs and hardware
- **User-Centric**: Voice-first interaction design
- **Robust**: Comprehensive error handling and graceful degradation
- **Documented**: Self-aware system with built-in documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for Whisper and GPT models
- Google for Gemini 2.5 models
- The Python community for excellent libraries
- Voice assistant research community

## 🔮 Roadmap

- [ ] **Phase 2**: Full voice assistant implementation
- [ ] **Phase 3**: Multi-language support
- [ ] **Phase 4**: Custom wake word training
- [ ] **Phase 5**: Plugin architecture
- [ ] **Phase 6**: Mobile app integration

---

**Ready to experience the future of voice interaction? Start talking to Jane! 🗣️✨**

For detailed documentation, see [DEMO_README.md](DEMO_README.md). 