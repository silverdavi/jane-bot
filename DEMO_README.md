# 🤖 Jane Voice Assistant - 10 Minute Demo with Background Transcription

**Complete end-to-end voice assistant with foreground conversation and background monitoring!**

## ✨ Features

This demo showcases the full voice assistant pipeline with dual-thread architecture:

### 🎯 **FOREGROUND Tasks (Main Thread)**
- 🎤 **Wake Word Detection** - Listen for "hey jane", "jane", "computer"
- 💬 **Conversation Management** - Question recording, processing, and responses
- 🧠 **Gemini 2.5 Intelligence** - Advanced question analysis and reasoning
- 🔊 **Natural Voice Responses** - OpenAI TTS with emotional context
- 📖 **Self-Aware Documentation** - Jane knows her own capabilities

### 🎧 **BACKGROUND Tasks (Separate Thread)**
- 📊 **Continuous Transcription** - Always listening and transcribing ambient audio
- 🔄 **Sliding Window Summaries** - 45-second intelligent summaries with context overlap
- 💾 **Persistent Buffer Management** - Automatic storage and cleanup
- 🛑 **Background Stop Detection** - Detects "STOP DEMO" commands

### 🎨 **Shared Features**
- 📚 **Smart Knowledge Base** - Automatic storage of conversations AND ambient summaries
- 🎨 **Rich Visual Feedback** - Colored state indicators and progress bars
- 🔊 **Audio Feedback** - Different tones for each state transition
- ⚡ **No Task Conflicts** - Foreground conversation doesn't interrupt background monitoring

## 🧠 Jane's Self-Knowledge

**Jane knows how she works!** She has comprehensive documentation in her knowledge base and can answer questions about:

### Questions You Can Ask Jane:
- **"What are your wake words?"**
- **"How long does recording last?"**
- **"What voices do you have?"**
- **"How does continuous transcription work?"**
- **"What are the stop commands?"**
- **"What audio feedback sounds do you make?"**
- **"How does the demo work?"**
- **"What files do you create?"**
- **"What AI models do you use?"**

### Documentation Available:
- `jane_help_documentation.txt` - Complete technical documentation
- `jane_quick_reference.txt` - Essential commands and settings
- `demo_session_*.txt` - Session logs with helpful tips

**Try asking Jane about herself - she's surprisingly knowledgeable about her own capabilities!**

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have:
- ✅ API keys configured in `config/api_keys.env`
- ✅ Microphone permissions enabled
- ✅ Speaker/headphones connected
- ✅ All dependencies installed

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python demo_voice_assistant.py
```

Press Enter when prompted to start the 10-minute demo!

## 🎯 Demo Flow

### States with Visual Indicators

1. **🔵 CONTINUOUS LISTENING** - Blue banner, transcribing everything continuously
2. **🟣 SLIDING WINDOW** - Purple banner, Gemini processing 45s contextual summaries  
3. **🟡 WAKE DETECTED** - Yellow banner, wake phrase recognized
4. **🟠 RECORDING** - Orange banner with voice activity detection
5. **🔴 PROCESSING** - Red banner, Gemini 2.5 analyzing
6. **🟣 SEARCHING** - Purple banner, knowledge base lookup
7. **🟢 GENERATING** - Green banner, response creation & polishing
8. **🎵 SPEAKING** - Cyan banner, audio playback
9. **🛑 STOP DETECTED** - Red banner, stop command recognized

### Audio Feedback Tones

- **Wake Detected**: High beep (800Hz)
- **Listening**: Low beep (400Hz)
- **Processing**: Medium beep (600Hz)
- **Responding**: High beep (1000Hz)
- **Error**: Low error tone (200Hz)

## 🎤 Usage Instructions

### Starting a Conversation

1. **Demo starts in "CONTINUOUS LISTENING" mode** - always transcribing
2. **Talk naturally** - everything is transcribed with intelligent 45s sliding window summaries
3. **Say wake word**: "Hey Jane" or "Computer" to enter conversation mode
4. **Wait for Yellow "WAKE DETECTED" confirmation**
5. **Ask your question naturally** - recording stops when you pause (2s silence)
6. **Watch real-time feedback** showing speaking/silence detection
7. **Listen to the response**
8. **Returns to continuous listening** - cycle repeats for 10 minutes
9. **Say "STOP DEMO" anytime** to end early (works during any state)

### Example Conversations

```
👤 "Hey Jane"
🤖 [Wake detected beep]

👤 "What's the weather like today?"
🤖 [Processing beep] → "I don't have access to real-time weather data, but I can help you find weather information if you'd like!"

👤 "Computer"
🤖 [Wake detected beep]

👤 "What can you help me with?"
🤖 [Processing beep] → "I can answer questions, help with information lookup, and remember our conversations in my knowledge base!"

👤 "Hey Jane"
🤖 [Wake detected beep]

👤 "What are your wake words?"
🤖 [Processing beep] → "My wake words are 'hey jane', 'hello jane', 'jane', and 'computer'. You can use any of these to start a conversation with me!"

👤 "Computer"
🤖 [Wake detected beep]

👤 "How long does recording last?"
🤖 [Processing beep] → "I record for up to 20 seconds maximum, but I'll stop automatically after 2 seconds of silence. I also won't wait more than 10 seconds for you to stop speaking."
```

## 🎨 Visual Elements

### State Banners
```
============================================================
🤖 JANE VOICE ASSISTANT | 14:30:25
📍 STATE: 🔵 LISTENING
💬 Waiting for wake word...
============================================================
```

### Progress Indicators
```
🎧 Continuous transcription... | Buffer: 23 | New: 8
🎧 Heard: 'what time is it'
📝 Processing sliding window summary (12 new segments)...
💾 Sliding window summary saved: ambient_sliding_summary_20241215_143210.txt
📄 Summary: User discussed dinner plans and inquired about current time, building on previous conversation about scheduling.
🧹 Cleaned up 15 old segments, kept 50 for context
🛑 STOP COMMAND detected: 'stop demo'!
```

During conversation mode:
```
🎤 Recording... speak now!
⏱️  2.3s | 🔊 SPEAKING | [██████████]
⏱️  3.1s | 🔇 SILENCE | [░░░░░░░░░░]
✅ Natural pause detected after 2.0s silence
🧠 Gemini 2.5 thinking...
🔄 Polishing response with GPT-4o-mini...
📚 Found 2 relevant documents
```

### Real-time Updates
```
⏱️  Demo time remaining: 8m 12s | Conversations: 2
🎧 Heard: 'hey jane what time is it'
🔄 Raw Gemini response: I can provide various types of assistance...
🔄 Polishing response with GPT-4o-mini...
💬 Final Response: "I don't have access to the current time, but I can help you with other questions!"
💾 Saved to KB: conversation_20241215_143045.txt
🛑 Say "STOP DEMO" to end early
```

## 🔧 Technical Details

### Audio Settings
- **Sample Rate**: 16kHz
- **Wake Detection**: 3-second chunks
- **Question Recording**: Advanced Voice Activity Detection
- **Silence Detection**: 2 seconds of silence stops recording
- **Maximum Silence Wait**: 10 seconds (prevents infinite waiting)
- **Maximum Recording**: 20 seconds per question
- **Audio Normalization**: Automatic level adjustment for better transcription
- **Latency Target**: <100ms for real-time feel

### API Models Used
- **Wake Word Detection**: OpenAI Whisper-1
- **Question Transcription**: OpenAI Whisper-1 (optimized with context)
- **Question Analysis**: Gemini 2.5 Flash
- **Complex Reasoning**: Gemini 2.5 Pro (when needed)
- **Response Polishing**: OpenAI GPT-4o-Mini (makes responses natural)
- **Speech Generation**: OpenAI GPT-4o-Mini-TTS

### Voice Selection
The demo automatically selects voices based on emotional tone:
- **Helpful**: Nova (default)
- **Friendly**: Alloy
- **Professional**: Echo
- **Enthusiastic**: Fable
- **Apologetic**: Onyx

## 💾 Knowledge Base

All conversations AND ambient summaries are automatically saved to `kb/user/`:

### Conversation Files
Example: `conversation_20241215_143045.txt`
```
Question: What's the weather like today?
Answer: I don't have access to real-time weather data, but I can help you find weather information if you'd like!
Timestamp: 2024-12-15T14:30:45.123456
```

### Sliding Window Summary Files  
Example: `ambient_sliding_summary_20241215_143210.txt`
```
Ambient Sliding Window Summary
==============================

Timestamp: 2024-12-15T14:32:10.789123
Window: segments 15-29
Duration: ~42 seconds
Total segments in window: 15
New segments processed: 10
Context overlap: 5 segments

Summary: User discussed dinner plans, mentioned needing to call mom, and inquired about time. This continues previous conversation about evening schedule coordination.

Raw Transcriptions:
==================
[CONTEXT] [2024-12-15T14:31:05] earlier we talked about meeting
[CONTEXT] [2024-12-15T14:31:08] so the plan was to coordinate
[NEW] [2024-12-15T14:32:05] i think we should have pizza for dinner
[NEW] [2024-12-15T14:32:07] oh wait i need to call mom later
[NEW] [2024-12-15T14:32:09] what time is it anyway
```

### Transcription Buffer
Example: `transcription_buffer.json`
```json
{
  "segments": [
    {
      "text": "i think we should have pizza",
      "timestamp": "2024-12-15T14:32:05.123456",
      "index": 23
    }
  ],
  "last_summarization_index": 20,
  "last_updated": "2024-12-15T14:32:10.789123"
}
```

## ⚠️ Troubleshooting

### No Wake Word Detection
- Speak louder and clearer
- Get closer to microphone
- Check microphone permissions
- Ensure `sounddevice` can access audio

### Poor Question Transcription
- Speak clearly and at normal pace
- Reduce background noise
- Ensure microphone is close (1-2 feet)
- Wait for voice activity indicator to show 🔊 SPEAKING
- Let natural pauses trigger recording stop

### Audio Issues
- Verify speaker/headphone connection
- Check system audio settings
- Ensure `afplay` (macOS) or audio player is available

### API Errors
- Verify API keys in `config/api_keys.env`
- Check internet connection
- Ensure sufficient API credits

### Performance Issues
- Close other audio applications
- Reduce background noise
- Use wired microphone if possible

## 📊 Demo Success Metrics

At the end of the demo, you'll see:
```
📊 Demo Summary:
  • Planned Duration: 10 minutes
  • Actual Duration: 8m 45s (if stopped early)
  • Conversations: 12
  • KB entries created: 23
  • State transitions: All working

🎉 SUCCESS! Voice assistant is fully functional!
```

## 🔄 Next Steps

After the demo, you can:
1. Explore the generated knowledge base files in `kb/user/`
2. Review the conversation history
3. Test individual components with `tests/run_tests.py`
4. Implement full application features from `python_implementation_plan.md`

## 🐛 Debug Mode

For detailed logging, run with Python's verbose flag:
```bash
python -v demo_voice_assistant.py
```

---

**Ready to experience the future of voice interaction? Run the 10-minute demo and start talking to Jane! 🚀** 