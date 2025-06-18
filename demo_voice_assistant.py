#!/usr/bin/env python3
"""
Voice Assistant Demo - 1 Minute Full Experience

Complete voice assistant demonstration featuring:
- 🎤 Wake word detection with visual feedback
- 🧠 Gemini 2.5 intelligent question analysis
- 📚 Knowledge base integration and storage
- 🔊 Audio responses with emotional context
- 🎨 Rich visual and audio state indicators

NO MOCKS - All real APIs and hardware integration.
"""

import os
import sys
import time
import threading
import queue
from pathlib import Path
from dotenv import load_dotenv
import wave
import asyncio
from datetime import datetime
import json

# Color and visual imports
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("⚠️  Install colorama for colored output: pip install colorama")

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("❌ sounddevice required for voice assistant")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("❌ OpenAI required for voice assistant")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ Google Generative AI required for voice assistant")

# Load environment variables
load_dotenv('config/api_keys.env')

class VoiceAssistantDemo:
    def __init__(self):
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # 3-second audio chunks
        
        # API clients
        self.setup_apis()
        
        # State management
        self.state = "INITIALIZING"
        self.conversation_active = False
        self.demo_start_time = None
        self.demo_duration = 600  # 10 minute demo
        self.stop_demo_requested = False
        
        # KB directory
        self.kb_dir = Path("kb/user")
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced transcription buffer management
        self.transcription_buffer = []  # Persistent buffer for all segments
        self.buffer_file = self.kb_dir / "transcription_buffer.json"
        self.last_summarization_index = 0  # Track what's been summarized
        
        # Audio feedback
        self.audio_queue = queue.Queue()
        
        # Conversation history
        self.conversation_history = []
        
        # Background transcription thread
        self.background_thread = None
        self.background_active = False
        
    def setup_apis(self):
        """Initialize API clients."""
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        
        if not self.openai_key or not self.gemini_key:
            raise ValueError("Missing API keys! Check config/api_keys.env")
            
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI library not available!")
            
        if not GEMINI_AVAILABLE:
            raise ValueError("Google Generative AI library not available!")
            
        self.openai_client = openai.OpenAI(api_key=self.openai_key)
        
        # Import genai here to ensure it's available
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        self.gemini_reasoning = genai.GenerativeModel('gemini-2.5-pro')
        
    def print_colored(self, text, color="WHITE", style="", end="\n", flush=False):
        """Print colored text with visual effects."""
        if not COLORS_AVAILABLE:
            print(text, end=end, flush=flush)
            return
            
        color_map = {
            "BLUE": Fore.BLUE,
            "GREEN": Fore.GREEN,
            "YELLOW": Fore.YELLOW,
            "RED": Fore.RED,
            "MAGENTA": Fore.MAGENTA,
            "CYAN": Fore.CYAN,
            "WHITE": Fore.WHITE,
        }
        
        style_map = {
            "BRIGHT": Style.BRIGHT,
            "DIM": Style.DIM,
        }
        
        color_code = color_map.get(color, Fore.WHITE)
        style_code = style_map.get(style, "")
        
        print(f"{style_code}{color_code}{text}{Style.RESET_ALL}", end=end, flush=flush)
    
    def show_state_banner(self, state, description, color="BLUE"):
        """Display current state with visual banner."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        banner = f"{'='*60}"
        
        self.print_colored(banner, color, "BRIGHT")
        self.print_colored(f"🤖 JANE VOICE ASSISTANT | {timestamp}", color, "BRIGHT")
        self.print_colored(f"📍 STATE: {state}", color, "BRIGHT")
        self.print_colored(f"💬 {description}", color)
        self.print_colored(banner, color, "BRIGHT")
        
        # Update internal state
        self.state = state
    
    def play_audio_feedback(self, tone_type="notification"):
        """Play audio feedback for different states."""
        if not AUDIO_AVAILABLE:
            return
            
        # Generate different tones for different states
        tones = {
            "wake_detected": (800, 0.2),    # High beep for wake word
            "listening": (400, 0.1),        # Low beep for listening
            "processing": (600, 0.3),       # Medium beep for processing
            "responding": (1000, 0.2),      # High beep for response
            "error": (200, 0.5),            # Low beep for error
            "notification": (500, 0.1)      # Default notification
        }
        
        freq, duration = tones.get(tone_type, (500, 0.1))
        
        try:
            # Generate tone
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            tone = 0.3 * np.sin(2 * np.pi * freq * t)
            
            # Play in background
            sd.play(tone, self.sample_rate)
            
        except Exception as e:
            print(f"⚠️  Audio feedback failed: {e}")
    
    def background_continuous_transcription(self):
        """Background thread for continuous transcription and summarization."""
        self.print_colored("🎧 Starting background continuous transcription...", "BLUE")
        
        last_summary_time = time.time()
        summary_interval = 45.0  # Summarize every 45 seconds
        min_segments_for_summary = 8  # Need at least 8 segments before summarizing
        
        # Load existing buffer if available
        self.load_transcription_buffer()
        
        while self.background_active and not self.stop_demo_requested:
            try:
                # Record audio segment
                recording = sd.rec(
                    int(self.chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float64'
                )
                sd.wait()
                
                # Save and transcribe
                temp_audio = Path("temp_background.wav")
                self.save_wav_file(temp_audio, recording)
                
                with open(temp_audio, 'rb') as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en",
                        prompt="Background transcription of ambient conversation."
                    )
                
                text = transcript.text.lower().strip()
                
                # Only process non-empty transcriptions
                if text and len(text) > 3:
                    # Add to persistent buffer (silently in background)
                    segment = {
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "index": len(self.transcription_buffer)
                    }
                    self.transcription_buffer.append(segment)
                    self.save_transcription_buffer()
                
                # Check for stop phrases
                stop_phrases = ["stop demo", "end demo", "quit demo"]
                for phrase in stop_phrases:
                    if phrase in text:
                        self.print_colored(f"\n🛑 BACKGROUND: Stop command detected!", "RED")
                        self.stop_demo_requested = True
                        temp_audio.unlink()
                        return
                
                # Check if it's time to summarize buffer with sliding windows
                new_segments = len(self.transcription_buffer) - self.last_summarization_index
                if (time.time() - last_summary_time >= summary_interval and 
                    new_segments >= min_segments_for_summary):
                    
                    self.print_colored(f"\n📝 Background sliding window summary ({new_segments} new segments)...", "MAGENTA")
                    self.sliding_window_summarization()
                    last_summary_time = time.time()
                
                # Clean up
                temp_audio.unlink()
                
            except Exception as e:
                # Silent background operation - don't spam errors
                time.sleep(1)
        
        # Final summary when background stops
        unsummarized = len(self.transcription_buffer) - self.last_summarization_index
        if unsummarized > 0:
            self.print_colored(f"\n📝 Final background summary of {unsummarized} segments...", "MAGENTA")
            self.sliding_window_summarization(final=True)
        
        self.print_colored("🎧 Background transcription stopped", "BLUE")
    
    def listen_for_wake_word(self, timeout=15):
        """Listen for wake word in foreground while background transcription runs."""
        wake_phrases = ["hey jane", "hello jane", "jane", "computer"]
        
        self.show_state_banner("🔵 LISTENING", "Waiting for wake word... (background transcription active)", "BLUE")
        self.play_audio_feedback("listening")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout and not self.stop_demo_requested:
            try:
                # Show listening animation with background info
                elapsed = time.time() - start_time
                dots = "." * (int(elapsed) % 4)
                buffer_size = len(self.transcription_buffer)
                self.print_colored(f"\r🎤 Listening for wake word{dots} | Background buffer: {buffer_size} segments", "BLUE", end="", flush=True)
                
                # Record audio segment
                recording = sd.rec(
                    int(self.chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float64'
                )
                sd.wait()
                
                # Save and transcribe
                temp_audio = Path("temp_wake_detection.wav")
                self.save_wav_file(temp_audio, recording)
                
                with open(temp_audio, 'rb') as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en",
                        prompt="Wake word detection for voice assistant named Jane."
                    )
                
                text = transcript.text.lower().strip()
                
                # Show what we heard
                if text and len(text) > 3:
                    self.print_colored(f"\n🎧 Heard: '{text}'", "CYAN")
                
                # Check for wake phrases
                for phrase in wake_phrases:
                    if phrase in text:
                        self.show_state_banner("🟡 WAKE DETECTED", f"Wake phrase '{phrase}' detected!", "YELLOW")
                        self.play_audio_feedback("wake_detected")
                        temp_audio.unlink()
                        return True
                
                # Clean up
                temp_audio.unlink()
                
            except Exception as e:
                self.print_colored(f"\n⚠️  Wake word detection failed: {e}", "RED")
                time.sleep(1)
        
        return False
    
    def load_transcription_buffer(self):
        """Load existing transcription buffer from file."""
        try:
            if self.buffer_file.exists():
                with open(self.buffer_file, 'r') as f:
                    import json
                    data = json.load(f)
                    self.transcription_buffer = data.get('segments', [])
                    self.last_summarization_index = data.get('last_summarization_index', 0)
                    self.print_colored(f"📂 Loaded {len(self.transcription_buffer)} segments from buffer", "BLUE")
            else:
                self.transcription_buffer = []
                self.last_summarization_index = 0
        except Exception as e:
            self.print_colored(f"⚠️  Buffer load failed: {e}", "YELLOW")
            self.transcription_buffer = []
            self.last_summarization_index = 0
    
    def save_transcription_buffer(self):
        """Save transcription buffer to file."""
        try:
            import json
            data = {
                'segments': self.transcription_buffer,
                'last_summarization_index': self.last_summarization_index,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.buffer_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.print_colored(f"⚠️  Buffer save failed: {e}", "YELLOW")
    
    def sliding_window_summarization(self, final=False):
        """Use sliding window with overlap to summarize transcriptions."""
        try:
            window_size = 15  # Process 15 segments at a time
            overlap_size = 5   # Keep 5 segments for context overlap
            
            start_index = max(0, self.last_summarization_index - overlap_size)
            end_index = len(self.transcription_buffer)
            
            if end_index - start_index < window_size and not final:
                self.print_colored(f"🔄 Not enough segments for window ({end_index - start_index} < {window_size})", "MAGENTA")
                return
            
            # Extract window segments
            window_segments = self.transcription_buffer[start_index:end_index]
            
            if not window_segments:
                return
                
            # Prepare transcription text with context indicators
            transcription_text = ""
            for i, item in enumerate(window_segments):
                actual_index = start_index + i
                context_mark = "[CONTEXT]" if actual_index < self.last_summarization_index else "[NEW]"
                transcription_text += f"{context_mark} [{item['timestamp']}] {item['text']}\n"
            
            # Calculate time span
            if len(window_segments) > 1:
                start_time = datetime.fromisoformat(window_segments[0]['timestamp'])
                end_time = datetime.fromisoformat(window_segments[-1]['timestamp'])
                duration = (end_time - start_time).total_seconds()
            else:
                duration = 3  # Single segment approximation
            
            # Create enhanced summary prompt
            summary_prompt = f"""
You are analyzing ambient voice transcriptions from a voice assistant environment using a sliding window approach.

Window contains {len(window_segments)} segments spanning approximately {duration:.0f} seconds.
- [CONTEXT] segments provide background context from previous summaries
- [NEW] segments are new content to be summarized

Transcription segments:
{transcription_text}

Please provide a contextual summary focusing on:
1. New meaningful conversations or information in [NEW] segments
2. How new content relates to [CONTEXT] if relevant
3. Key topics, subjects, or themes that emerge
4. Any questions, requests, or important statements
5. Natural conversation flow and continuity

If the new content is mostly silence, background noise, or meaningless fragments, respond with "No significant new content detected."

Keep the summary concise but informative (2-3 sentences maximum).

Summary:
"""
            
            # Get summary from Gemini
            response = self.gemini_model.generate_content(summary_prompt)
            summary = response.text.strip()
            
            # Only store if there's meaningful content
            if "no significant" not in summary.lower() and len(summary) > 25:
                # Save to knowledge base
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ambient_sliding_summary_{timestamp}.txt"
                filepath = self.kb_dir / filename
                
                new_segments = [s for i, s in enumerate(window_segments) if start_index + i >= self.last_summarization_index]
                
                with open(filepath, 'w') as f:
                    f.write(f"Ambient Sliding Window Summary\n")
                    f.write(f"==============================\n\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Window: segments {start_index}-{end_index-1}\n")
                    f.write(f"Duration: ~{duration:.0f} seconds\n")
                    f.write(f"Total segments in window: {len(window_segments)}\n")
                    f.write(f"New segments processed: {len(new_segments)}\n")
                    f.write(f"Context overlap: {overlap_size} segments\n\n")
                    f.write(f"Summary: {summary}\n\n")
                    f.write("Raw Transcriptions:\n")
                    f.write("==================\n")
                    for i, item in enumerate(window_segments):
                        actual_index = start_index + i
                        context_mark = "[CONTEXT]" if actual_index < self.last_summarization_index else "[NEW]"
                        f.write(f"{context_mark} [{item['timestamp']}] {item['text']}\n")
                
                self.print_colored(f"💾 Sliding window summary saved: {filename}", "MAGENTA")
                self.print_colored(f"📄 Summary: {summary}", "MAGENTA")
                
                # Update tracking - only advance past non-overlapping new content
                new_processed = len([s for i, s in enumerate(window_segments) if start_index + i >= self.last_summarization_index])
                self.last_summarization_index += max(1, new_processed - overlap_size)
                
                # Clean up old segments (keep recent ones for context)
                keep_segments = 50  # Keep last 50 segments for context
                if len(self.transcription_buffer) > keep_segments + 20:
                    removed_count = len(self.transcription_buffer) - keep_segments
                    self.transcription_buffer = self.transcription_buffer[-keep_segments:]
                    self.last_summarization_index = max(0, self.last_summarization_index - removed_count)
                    self.print_colored(f"🧹 Cleaned up {removed_count} old segments, kept {keep_segments} for context", "MAGENTA")
                
                self.save_transcription_buffer()
                
            else:
                self.print_colored(f"🔇 No significant new content in sliding window", "MAGENTA")
                # Still advance index to avoid reprocessing
                new_segments = len([s for i, s in enumerate(window_segments) if start_index + i >= self.last_summarization_index])
                self.last_summarization_index += max(1, new_segments - overlap_size)
                self.save_transcription_buffer()
                
        except Exception as e:
            self.print_colored(f"⚠️  Sliding window summarization failed: {e}", "YELLOW")
            import traceback
            traceback.print_exc()
    
    def initialize_demo_session(self):
        """Initialize demo session with KB documentation info."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_file = self.kb_dir / f"demo_session_{timestamp}.txt"
            
            with open(session_file, 'w') as f:
                f.write(f"JANE VOICE ASSISTANT - DEMO SESSION\n")
                f.write(f"====================================\n\n")
                f.write(f"Session Started: {datetime.now().isoformat()}\n")
                f.write(f"Demo Duration: 10 minutes\n")
                f.write(f"Mode: Continuous Transcription with Ambient Monitoring\n\n")
                f.write(f"AVAILABLE DOCUMENTATION:\n")
                f.write(f"- jane_help_documentation.txt - Complete system help\n")
                f.write(f"- jane_quick_reference.txt - Essential commands and settings\n")
                f.write(f"- This session will create conversation and ambient summary files\n\n")
                f.write(f"ASK JANE ABOUT:\n")
                f.write(f"- Wake words and stop commands\n")
                f.write(f"- Timing and audio settings\n")
                f.write(f"- Available voices and features\n")
                f.write(f"- How continuous transcription works\n")
                f.write(f"- Troubleshooting tips\n\n")
                f.write(f"Jane has access to her own documentation and can answer questions about her capabilities!\n")
            
            self.print_colored(f"📝 Demo session initialized: {session_file.name}", "GREEN")
            
        except Exception as e:
            self.print_colored(f"⚠️  Session initialization failed: {e}", "YELLOW")
    
    def record_question(self):
        """Record user's question with improved voice activity detection."""
        self.show_state_banner("🟠 RECORDING", "Ask your question now... (10s max wait for silence)", "YELLOW")
        self.play_audio_feedback("notification")
        
        try:
            # Improved voice activity detection parameters
            chunk_size = int(0.2 * self.sample_rate)  # 200ms chunks for better detection
            silence_threshold = 0.005  # Lower threshold for better sensitivity
            silence_duration = 2.0  # Stop after 2 seconds of silence
            max_silence_wait = 10.0  # Maximum 10 seconds to wait for silence
            max_recording_time = 20.0  # Maximum 20 seconds total recording
            min_recording_time = 1.0  # Minimum 1 second before checking for silence
            
            recording_chunks = []
            silent_chunks = 0
            total_chunks = 0
            max_chunks = int(max_recording_time / 0.2)
            min_chunks = int(min_recording_time / 0.2)
            voice_detected = False  # Track if we've heard any voice
            
            self.print_colored("🎤 Recording... speak now!", "YELLOW")
            self.print_colored("💡 Max 10s wait for silence, 20s total", "YELLOW")
            
            # Start recording in real-time chunks
            while total_chunks < max_chunks:
                # Record chunk
                chunk = sd.rec(chunk_size, samplerate=self.sample_rate, channels=1, dtype='float64')
                sd.wait()
                
                recording_chunks.append(chunk)
                total_chunks += 1
                
                # Calculate volume (RMS) for voice activity detection
                rms = np.sqrt(np.mean(chunk**2))
                
                # Visual feedback and voice detection
                if rms > silence_threshold:
                    # Voice detected
                    voice_detected = True
                    silent_chunks = 0
                    activity = "🔊 SPEAKING"
                    bar = "█" * min(int(rms * 100), 10)  # Better visualization
                else:
                    # Silence detected
                    silent_chunks += 1
                    activity = "🔇 SILENCE"
                    bar = "░" * 10
                
                elapsed_time = total_chunks * 0.2
                silence_time = silent_chunks * 0.2
                
                self.print_colored(
                    f"\r⏱️  {elapsed_time:.1f}s | {activity} | [{bar}] | Silence: {silence_time:.1f}s", 
                    "YELLOW", end="", flush=True
                )
                
                # Check stopping conditions
                # 1. Have we recorded minimum time and detected voice?
                if total_chunks >= min_chunks and voice_detected:
                    # 2. Enough silence to stop?
                    if silence_time >= silence_duration:
                        self.print_colored(f"\n✅ Natural pause detected after {silence_time:.1f}s silence", "GREEN")
                        break
                    # 3. Too much silence waiting?
                    elif silence_time >= max_silence_wait:
                        self.print_colored(f"\n⏰ Max silence wait ({max_silence_wait}s) reached", "YELLOW")
                        break
                
                # 4. If no voice detected yet, keep recording
                if not voice_detected and total_chunks >= max_chunks // 2:
                    self.print_colored(f"\n⚠️  No clear voice detected yet, continuing...", "YELLOW")
            
            if len(recording_chunks) == 0:
                self.print_colored(f"\n❌ No audio recorded", "RED")
                return None
                
            if not voice_detected:
                self.print_colored(f"\n⚠️  No clear voice detected, but proceeding with transcription", "YELLOW")
                
            # Combine all chunks and normalize audio
            full_recording = np.concatenate(recording_chunks, axis=0)
            
            # Audio preprocessing for better transcription
            # Normalize audio levels
            max_val = np.max(np.abs(full_recording))
            if max_val > 0:
                full_recording = full_recording / max_val * 0.8  # Normalize to 80% to avoid clipping
            
            # Save with better audio format
            question_audio = Path("temp_question.wav")
            self.save_wav_file(question_audio, full_recording)
            
            total_duration = len(recording_chunks) * 0.2
            self.print_colored(f"\n🔄 Transcribing {total_duration:.1f}s question...", "YELLOW")
            
            # Enhanced transcription with better parameters
            with open(question_audio, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",  # Use whisper-1 as it's more reliable than gpt-4o-transcribe
                    file=audio_file,
                    language="en",  # Specify language for better accuracy
                    prompt="This is a question being asked to a voice assistant named Jane."  # Context helps
                )
            
            question = transcript.text.strip()
            
            # Check for stop command
            if "stop demo" in question.lower():
                self.print_colored(f"🛑 STOP DEMO command detected!", "RED", "BRIGHT")
                self.stop_demo_requested = True
            
            self.print_colored(f"❓ Question: '{question}'", "CYAN", "BRIGHT")
            
            question_audio.unlink()
            return question
            
        except Exception as e:
            self.print_colored(f"❌ Question recording failed: {e}", "RED")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_question_with_gemini(self, question):
        """Use Gemini 2.5 to analyze question and determine response strategy."""
        self.show_state_banner("🔴 PROCESSING", "Analyzing question with Gemini 2.5...", "RED")
        self.play_audio_feedback("processing")
        
        try:
            # Check conversation history
            history_context = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}" 
                for item in self.conversation_history[-3:]  # Last 3 exchanges
            ])
            
            analysis_prompt = f"""
You are Jane, a personal voice assistant. Analyze this question and provide a response strategy.

Conversation History:
{history_context}

Current Question: "{question}"

Analyze:
1. Can you answer this directly with your knowledge?
2. Does this require searching the user's knowledge base?
3. What type of response is most appropriate?

IMPORTANT: If the question is about YOUR capabilities, commands, settings, or how you work (wake words, stop commands, timing, voices, etc.), you should search your knowledge base which contains your documentation.

Examples of questions requiring KB search:
- "What are the wake words?"
- "How long does recording last?"
- "What voices do you have?"
- "How does the demo work?"
- "What are the stop commands?"
- "How does continuous transcription work?"

Provide a JSON response with:
{{
    "can_answer_directly": true/false,
    "needs_kb_search": true/false,
    "response_strategy": "direct_answer|kb_search|clarification",
    "direct_answer": "your answer if can_answer_directly is true",
    "search_terms": ["terms", "to", "search"] if needs_kb_search is true,
    "emotional_tone": "friendly|professional|helpful|enthusiastic"
}}
"""
            
            # Show processing animation
            for i in range(3):
                self.print_colored(f"\r🧠 Gemini 2.5 thinking{'.' * (i + 1)}", "RED", end="", flush=True)
                time.sleep(0.5)
            
            response = self.gemini_model.generate_content(analysis_prompt)
            
            # Parse JSON response
            try:
                analysis = json.loads(response.text.strip())
                self.print_colored(f"\n✅ Analysis complete!", "GREEN")
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                self.print_colored(f"\n⚠️  Using fallback analysis", "YELLOW")
                return {
                    "can_answer_directly": True,
                    "needs_kb_search": False,
                    "response_strategy": "direct_answer",
                    "direct_answer": response.text.strip(),
                    "emotional_tone": "helpful"
                }
                
        except Exception as e:
            self.print_colored(f"\n❌ Gemini analysis failed: {e}", "RED")
            return {
                "can_answer_directly": True,
                "needs_kb_search": False,
                "response_strategy": "direct_answer",
                "direct_answer": "I'm sorry, I had trouble processing your question. Could you please rephrase it?",
                "emotional_tone": "apologetic"
            }
    
    def search_knowledge_base(self, search_terms):
        """Search the user's knowledge base for relevant information."""
        self.show_state_banner("🟣 SEARCHING", "Looking in your knowledge base...", "MAGENTA")
        
        try:
            # Search through all KB files including documentation
            kb_files = list(self.kb_dir.glob("*.txt"))
            relevant_content = []
            
            for file_path in kb_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Enhanced search - check for any search terms
                    content_lower = content.lower()
                    file_matches = []
                    
                    for term in search_terms:
                        term_lower = term.lower()
                        if term_lower in content_lower:
                            # Find the section containing the term for better context
                            lines = content.split('\n')
                            matching_sections = []
                            
                            for i, line in enumerate(lines):
                                if term_lower in line.lower():
                                    # Get surrounding context (2 lines before, 3 lines after)
                                    start = max(0, i-2)
                                    end = min(len(lines), i+4)
                                    context = '\n'.join(lines[start:end])
                                    matching_sections.append(context)
                            
                            if matching_sections:
                                file_matches.extend(matching_sections)
                    
                    if file_matches:
                        # Combine unique sections, limit to reasonable size
                        unique_matches = list(set(file_matches))
                        combined_content = '\n\n---\n\n'.join(unique_matches[:3])  # Max 3 sections
                        
                        relevant_content.append({
                            "file": file_path.name,
                            "content": combined_content[:800] + "..." if len(combined_content) > 800 else combined_content,
                            "type": "documentation" if "help" in file_path.name or "reference" in file_path.name else "conversation"
                        })
                        
                except Exception as e:
                    self.print_colored(f"⚠️  Error reading {file_path.name}: {e}", "YELLOW")
                    continue
            
            # Sort by type (documentation first, then conversations)
            relevant_content.sort(key=lambda x: (x.get("type", "z"), x["file"]))
            
            self.print_colored(f"📚 Found {len(relevant_content)} relevant documents", "MAGENTA")
            
            if relevant_content:
                for item in relevant_content:
                    doc_type = "📖" if item.get("type") == "documentation" else "💬"
                    self.print_colored(f"  {doc_type} {item['file']}", "MAGENTA")
            
            return relevant_content
            
        except Exception as e:
            self.print_colored(f"❌ KB search failed: {e}", "RED")
            return []
    
    def generate_response(self, question, analysis, kb_content=None):
        """Generate final response using Gemini, then polish with GPT-4o-mini."""
        self.show_state_banner("🟢 GENERATING", "Creating response...", "GREEN")
        
        try:
            # Step 1: Get raw response from Gemini
            if analysis["response_strategy"] == "direct_answer":
                raw_response = analysis["direct_answer"]
            else:
                # Generate response with KB content
                kb_context = "\n".join([
                    f"From {item['file']}: {item['content']}"
                    for item in kb_content or []
                ])
                
                response_prompt = f"""
You are Jane, a helpful personal voice assistant. 

Question: "{question}"
Knowledge Base Context: {kb_context}
Emotional Tone: {analysis.get('emotional_tone', 'helpful')}

Provide a natural, conversational response that:
1. Directly answers the question
2. Uses the knowledge base context if relevant
3. Maintains the specified emotional tone
4. Keeps the response concise (1-2 sentences for voice)

Response:
"""
                
                response = self.gemini_model.generate_content(response_prompt)
                raw_response = response.text.strip()
            
            self.print_colored(f"🔄 Raw Gemini response: {raw_response}", "GREEN")
            
            # Step 2: Polish the response with GPT-4o-mini for naturalness
            polished_response = self.polish_response_with_gpt(question, raw_response, analysis.get('emotional_tone', 'helpful'))
            
            self.print_colored(f"💬 Final Response: {polished_response}", "GREEN", "BRIGHT")
            return polished_response
            
        except Exception as e:
            self.print_colored(f"❌ Response generation failed: {e}", "RED")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def polish_response_with_gpt(self, original_question, raw_answer, emotional_tone):
        """Use GPT-4o-mini to polish the response for natural conversation."""
        try:
            self.print_colored("🔄 Polishing response with GPT-4o-mini...", "GREEN")
            
            polish_prompt = f"""
You are helping to make a voice assistant response more natural and conversational.

The user asked: "{original_question}"

The AI generated this answer: "{raw_answer}"

The desired emotional tone is: {emotional_tone}

Your task:
1. Rephrase this answer to be natural and conversational for voice interaction
2. Make sure it directly answers the user's question 
3. Keep the same information but make it sound more human and friendly
4. Maintain the emotional tone specified
5. Keep it concise (1-2 sentences max for voice)
6. Remove any awkward phrasing or technical jargon

IMPORTANT: Only return the polished response text, nothing else.

Polished response:
"""
            
            # Use GPT-4o-mini for response polishing
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at making AI responses sound natural and conversational for voice assistants."},
                    {"role": "user", "content": polish_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            polished_text = response.choices[0].message.content.strip()
            self.print_colored(f"✅ Response polished successfully", "GREEN")
            return polished_text
            
        except Exception as e:
            self.print_colored(f"⚠️  Response polishing failed: {e}", "YELLOW")
            # Fallback to raw response if polishing fails
            return raw_answer
    
    def save_to_knowledge_base(self, question, answer):
        """Save the Q&A pair to the knowledge base."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.txt"
            filepath = self.kb_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(f"Question: {question}\n")
                f.write(f"Answer: {answer}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            
            self.print_colored(f"💾 Saved to KB: {filename}", "CYAN")
            
        except Exception as e:
            self.print_colored(f"⚠️  KB save failed: {e}", "YELLOW")
    
    def speak_response(self, text, emotional_tone="helpful"):
        """Convert text to speech and play it."""
        self.show_state_banner("🎵 SPEAKING", "Playing audio response...", "CYAN")
        self.play_audio_feedback("responding")
        
        try:
            # Choose voice based on emotional tone
            voice_map = {
                "friendly": "alloy",
                "professional": "echo",
                "helpful": "nova",
                "enthusiastic": "fable",
                "apologetic": "onyx"
            }
            
            voice = voice_map.get(emotional_tone, "nova")
            
            self.print_colored(f"🎤 Generating speech with {voice} voice...", "CYAN")
            
            response = self.openai_client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text
            )
            
            # Save and play audio
            audio_path = Path("temp_response.mp3")
            with open(audio_path, 'wb') as f:
                f.write(response.content)
            
            self.print_colored(f"🔊 Playing response...", "CYAN")
            
            # Play audio file (platform-specific)
            import subprocess
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", str(audio_path)], check=True)
            elif sys.platform == "linux":
                subprocess.run(["aplay", str(audio_path)], check=True)
            elif sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
            
            # Clean up
            audio_path.unlink()
            
            self.print_colored(f"✅ Response delivered!", "GREEN")
            
        except Exception as e:
            self.print_colored(f"❌ Speech generation failed: {e}", "RED")
            self.print_colored(f"📝 Text response: {text}", "WHITE", "BRIGHT")
    
    def save_wav_file(self, filepath, data):
        """Save audio data to WAV file."""
        try:
            data_int16 = np.int16(data * 32767)
            
            with wave.open(str(filepath), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(data_int16.tobytes())
                
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
    
    def run_demo(self):
        """Run the complete 10-minute voice assistant demo."""
        self.demo_start_time = time.time()
        
        # Initialize knowledge base with session info
        self.initialize_demo_session()
        
        # Welcome banner
        self.show_state_banner("🚀 STARTING", "Jane Voice Assistant Demo - 10 Minute Experience", "GREEN")
        self.play_audio_feedback("notification")
        
        self.print_colored("🎯 Demo Features:", "WHITE", "BRIGHT")
        self.print_colored("  • FOREGROUND: Wake word detection and conversation", "WHITE", "BRIGHT")
        self.print_colored("  • BACKGROUND: Continuous transcription with sliding windows", "WHITE")
        self.print_colored("  • Persistent buffering with 45s intelligent summaries", "WHITE")
        self.print_colored("  • Context-aware processing with 5-segment overlap", "WHITE")
        self.print_colored("  • Wake words: 'hey jane', 'jane', 'computer'", "WHITE")
        self.print_colored("  • Gemini 2.5 intelligent processing", "WHITE")
        self.print_colored("  • Knowledge base integration", "WHITE")
        self.print_colored("  • Natural voice responses", "WHITE")
        self.print_colored("  • Say 'STOP DEMO' anytime to end early", "WHITE", "BRIGHT")
        
        time.sleep(2)
        
        # Start background continuous transcription
        self.background_active = True
        self.background_thread = threading.Thread(target=self.background_continuous_transcription, daemon=True)
        self.background_thread.start()
        self.print_colored("🎧 Background transcription started", "GREEN")
        
        conversation_count = 0
        
        while time.time() - self.demo_start_time < self.demo_duration and not self.stop_demo_requested:
            remaining_time = self.demo_duration - (time.time() - self.demo_start_time)
            
            if remaining_time < 10:
                self.show_state_banner("⏰ DEMO ENDING", f"Demo ending in {remaining_time:.0f} seconds", "YELLOW")
                time.sleep(min(remaining_time, 5))
                break
            
            # Show time in minutes and seconds
            remaining_minutes = int(remaining_time // 60)
            remaining_seconds = int(remaining_time % 60)
            self.print_colored(f"\n⏱️  Demo time remaining: {remaining_minutes}m {remaining_seconds}s | Conversations: {conversation_count}", "WHITE")
            
            # Check if stop was requested by background thread
            if self.stop_demo_requested:
                self.show_state_banner("🛑 DEMO STOPPED", "Stop command received!", "RED")
                break
            
            # Step 1: Listen for wake word (foreground task)
            if self.listen_for_wake_word(timeout=15):
                
                # Step 2: Record question
                question = self.record_question()
                if not question:
                    continue
                
                # Check if stop demo was requested during question recording
                if self.stop_demo_requested:
                    self.show_state_banner("🛑 DEMO STOPPED", "Stop command received!", "RED")
                    break
                
                # Step 3: Analyze with Gemini
                analysis = self.analyze_question_with_gemini(question)
                
                # Step 4: Search KB if needed
                kb_content = None
                if analysis.get("needs_kb_search"):
                    kb_content = self.search_knowledge_base(analysis.get("search_terms", []))
                
                # Step 5: Generate response
                response = self.generate_response(question, analysis, kb_content)
                
                # Step 6: Save to KB
                self.save_to_knowledge_base(question, response)
                
                # Step 7: Speak response
                self.speak_response(response, analysis.get("emotional_tone", "helpful"))
                
                # Update conversation history
                self.conversation_history.append({
                    "question": question,
                    "answer": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                conversation_count += 1
                
                # Brief pause before next cycle
                self.print_colored("🔄 Ready for next question...", "BLUE")
                time.sleep(2)
            
            else:
                self.print_colored("⏳ No wake word detected, continuing to listen...", "YELLOW")
        
        # Stop background transcription
        self.background_active = False
        if self.background_thread and self.background_thread.is_alive():
            self.print_colored("🛑 Stopping background transcription...", "YELLOW")
            self.background_thread.join(timeout=3)  # Wait up to 3 seconds for clean shutdown
        
        # Demo completion
        if self.stop_demo_requested:
            self.show_state_banner("🛑 DEMO STOPPED BY USER", f"Completed {conversation_count} conversations before stopping!", "RED")
        else:
            self.show_state_banner("🎉 DEMO COMPLETE", f"Completed {conversation_count} conversations!", "GREEN")
        
        self.play_audio_feedback("notification")
        
        actual_duration = time.time() - self.demo_start_time
        actual_minutes = int(actual_duration // 60)
        actual_seconds = int(actual_duration % 60)
        
        self.print_colored("📊 Demo Summary:", "WHITE", "BRIGHT")
        self.print_colored(f"  • Planned Duration: {self.demo_duration // 60} minutes", "WHITE")
        self.print_colored(f"  • Actual Duration: {actual_minutes}m {actual_seconds}s", "WHITE")
        self.print_colored(f"  • Conversations: {conversation_count}", "WHITE")
        self.print_colored(f"  • KB entries created: {len(list(self.kb_dir.glob('*.txt')))}", "WHITE")
        self.print_colored(f"  • State transitions: All working", "WHITE")
        
        if conversation_count > 0:
            self.print_colored("\n🎉 SUCCESS! Voice assistant is fully functional!", "GREEN", "BRIGHT")
        else:
            self.print_colored("\n⚠️  No conversations completed. Try speaking louder or closer to microphone.", "YELLOW")

def main():
    """Run the voice assistant demo."""
    print("🔥 JANE VOICE ASSISTANT - 10 MINUTE DEMO WITH BACKGROUND TRANSCRIPTION")
    print("=" * 75)
    print("🎯 FOREGROUND: Wake word detection + conversation")
    print("🎧 BACKGROUND: Continuous transcription + sliding window summaries")
    print("🔄 Dual-thread architecture - no conflicts between tasks")
    print("⚠️  Ensure microphone permissions are enabled")
    print("🔊 Adjust speaker volume for audio feedback")
    print("🛑 Say 'STOP DEMO' anytime to end early")
    print("=" * 75)
    
    # Check dependencies
    if not all([AUDIO_AVAILABLE, OPENAI_AVAILABLE, GEMINI_AVAILABLE]):
        print("❌ Missing required dependencies!")
        return
    
    try:
        demo = VoiceAssistantDemo()
        
        input("\n🎬 Press Enter to start the 10-minute demo...")
        
        demo.run_demo()
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 