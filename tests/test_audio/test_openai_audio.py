#!/usr/bin/env python3
"""
Test OpenAI Audio Functionality

Tests REAL OpenAI API calls for:
- Speech-to-text transcription
- Text-to-speech generation
- Audio quality and performance
- Different voice options

NO MOCKS - All real API calls.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import openai
import tempfile
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
load_dotenv('config/api_keys.env')

class OpenAIAudioTester:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        self.client = openai.OpenAI(api_key=self.openai_key)
        
        # Test audio directory
        self.audio_test_dir = Path("tests/test_audio/generated")
        self.audio_test_dir.mkdir(parents=True, exist_ok=True)
        
    def test_text_to_speech(self):
        """Test OpenAI TTS with different voices and content."""
        print("🎤 Testing Text-to-Speech Generation...")
        
        test_cases = [
            {
                "text": "Hello! I'm your voice assistant Jane. I'm ready to help you with information and tasks.",
                "voice": "alloy",
                "filename": "greeting_alloy.mp3"
            },
            {
                "text": "The weather today is sunny with a high of 75 degrees. Perfect for outdoor activities!",
                "voice": "echo",
                "filename": "weather_echo.mp3"
            },
            {
                "text": "I found 3 documents related to your voice assistant project. Would you like me to summarize them?",
                "voice": "fable",
                "filename": "documents_fable.mp3"
            },
            {
                "text": "Error: Unable to connect to the knowledge base. Please check your configuration.",
                "voice": "onyx",
                "filename": "error_onyx.mp3"
            },
            {
                "text": "Congratulations! Your voice assistant setup is now complete and ready for use.",
                "voice": "nova",
                "filename": "success_nova.mp3"
            }
        ]
        
        generated_files = []
        
        for test_case in test_cases:
            try:
                print(f"\n  🔊 Generating: {test_case['voice']} voice")
                print(f"     Text: {test_case['text'][:50]}...")
                
                start_time = time.time()
                
                response = self.client.audio.speech.create(
                    model="tts-1-hd",  # High quality model
                    voice=test_case['voice'],
                    input=test_case['text']
                )
                
                generation_time = time.time() - start_time
                
                # Save audio file
                audio_path = self.audio_test_dir / test_case['filename']
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(audio_path)
                print(f"     ✅ Generated in {generation_time:.2f}s, Size: {file_size:,} bytes")
                
                generated_files.append(audio_path)
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                return False, []
                
        return True, generated_files
    
    def test_steerable_tts(self):
        """Test the new steerable TTS model."""
        print("\n🎭 Testing Steerable TTS (gpt-4o-mini-tts)...")
        
        steerable_tests = [
            {
                "text": "Welcome to your personal assistant! I'm excited to help you today.",
                "style": "excited and enthusiastic",
                "filename": "steerable_excited.mp3"
            },
            {
                "text": "I'm sorry, but I couldn't find the information you requested.",
                "style": "apologetic and understanding",
                "filename": "steerable_apologetic.mp3"
            },
            {
                "text": "The system is processing your request. Please wait a moment.",
                "style": "calm and professional",
                "filename": "steerable_professional.mp3"
            }
        ]
        
        for test_case in steerable_tests:
            try:
                print(f"\n  🎪 Style: {test_case['style']}")
                print(f"     Text: {test_case['text']}")
                
                # Use the steerable model with style instruction
                enhanced_text = f"[Speaking in a {test_case['style']} manner] {test_case['text']}"
                
                start_time = time.time()
                
                response = self.client.audio.speech.create(
                    model="gpt-4o-mini-tts",  # Steerable model
                    voice="alloy",
                    input=enhanced_text
                )
                
                generation_time = time.time() - start_time
                
                # Save audio file
                audio_path = self.audio_test_dir / test_case['filename']
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(audio_path)
                print(f"     ✅ Generated in {generation_time:.2f}s, Size: {file_size:,} bytes")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                return False
                
        return True
    
    def test_transcription_models(self):
        """Test transcription with created audio files."""
        print("\n📝 Testing Speech-to-Text Transcription...")
        
        # Create a test audio file first
        test_text = "This is a test of the speech-to-text transcription system. The quick brown fox jumps over the lazy dog."
        
        try:
            # Generate test audio
            print("  🎧 Creating test audio for transcription...")
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=test_text
            )
            
            test_audio_path = self.audio_test_dir / "test_transcription.mp3"
            with open(test_audio_path, 'wb') as f:
                f.write(response.content)
            
            print(f"     ✅ Test audio created: {test_audio_path}")
            
            # Test transcription models
            transcription_models = [
                "gpt-4o-transcribe",
                "gpt-4o-mini-transcribe", 
                "whisper-1"
            ]
            
            for model in transcription_models:
                try:
                    print(f"\n  🔍 Testing {model}...")
                    print(f"     Original: {test_text}")
                    
                    start_time = time.time()
                    
                    with open(test_audio_path, 'rb') as audio_file:
                        transcript = self.client.audio.transcriptions.create(
                            model=model,
                            file=audio_file
                        )
                    
                    transcription_time = time.time() - start_time
                    
                    print(f"     Transcribed: {transcript.text}")
                    print(f"     ✅ Completed in {transcription_time:.2f}s")
                    
                    # Simple accuracy check
                    original_words = set(test_text.lower().split())
                    transcribed_words = set(transcript.text.lower().split())
                    accuracy = len(original_words & transcribed_words) / len(original_words) * 100
                    print(f"     📊 Word accuracy: {accuracy:.1f}%")
                    
                except Exception as e:
                    print(f"     ❌ {model} failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"  ❌ Transcription test setup failed: {e}")
            return False
    
    def test_real_time_performance(self):
        """Test performance metrics for real-time use."""
        print("\n⚡ Testing Real-time Performance...")
        
        # Test short responses (typical for voice assistant)
        short_texts = [
            "Yes",
            "I understand",
            "Let me check that for you",
            "Here's what I found",
            "Would you like me to continue?"
        ]
        
        performance_data = []
        
        for text in short_texts:
            try:
                start_time = time.time()
                
                response = self.client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="alloy",
                    input=text
                )
                
                generation_time = time.time() - start_time
                file_size = len(response.content)
                
                performance_data.append({
                    'text': text,
                    'chars': len(text),
                    'time': generation_time,
                    'size': file_size,
                    'chars_per_sec': len(text) / generation_time
                })
                
                print(f"  📊 '{text}' -> {generation_time:.3f}s ({len(text)} chars)")
                
            except Exception as e:
                print(f"  ❌ Performance test failed: {e}")
                return False
        
        # Calculate averages
        avg_time = sum(p['time'] for p in performance_data) / len(performance_data)
        avg_chars_per_sec = sum(p['chars_per_sec'] for p in performance_data) / len(performance_data)
        
        print(f"\n📈 Performance Summary:")
        print(f"  Average generation time: {avg_time:.3f}s")
        print(f"  Average processing speed: {avg_chars_per_sec:.1f} chars/sec")
        print(f"  Target for real-time: <5s ✅" if avg_time < 5 else "  Target for real-time: <5s ❌")
        
        return True
    
    def test_audio_formats(self):
        """Test different audio format outputs."""
        print("\n🎵 Testing Audio Formats...")
        
        test_text = "Testing audio format compatibility."
        
        # Note: OpenAI TTS currently outputs MP3, but let's test the quality
        try:
            response = self.client.audio.speech.create(
                model="tts-1-hd",
                voice="alloy",
                input=test_text,
                response_format="mp3"
            )
            
            audio_path = self.audio_test_dir / "format_test.mp3"
            with open(audio_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(audio_path)
            print(f"  ✅ MP3 format: {file_size:,} bytes")
            
            # Test if we can read basic audio properties
            try:
                import wave
                # Note: MP3 requires different library, but we can check file existence
                if audio_path.exists():
                    print(f"  ✅ Audio file created successfully")
                    return True
            except ImportError:
                print(f"  ⚠️  Wave library not available, but file created")
                return True
                
        except Exception as e:
            print(f"  ❌ Audio format test failed: {e}")
            return False
    
    def list_generated_files(self):
        """List all generated audio files for manual testing."""
        print("\n📁 Generated Audio Files for Manual Testing:")
        
        if not self.audio_test_dir.exists():
            print("  No audio files generated")
            return
            
        audio_files = list(self.audio_test_dir.glob("*.mp3"))
        
        if not audio_files:
            print("  No audio files found")
            return
            
        for audio_file in sorted(audio_files):
            file_size = os.path.getsize(audio_file)
            print(f"  🎵 {audio_file.name} ({file_size:,} bytes)")
        
        print(f"\n💡 To test audio playback:")
        print(f"   cd {self.audio_test_dir}")
        print(f"   open *.mp3  # (on macOS)")
        print(f"   # or use your preferred audio player")
    
    def cleanup(self):
        """Clean up test files (optional)."""
        print("\n🧹 Audio files preserved for manual testing")
        print(f"   Location: {self.audio_test_dir}")
        print("   Run 'rm tests/test_audio/generated/*.mp3' to clean up")

def main():
    print("🔥 OPENAI AUDIO TESTING")
    print("=" * 50)
    print("Testing with REAL OpenAI API calls (NO MOCKS)")
    print("=" * 50)
    
    tester = OpenAIAudioTester()
    
    try:
        # Run all tests
        tests = [
            ("Text-to-Speech", lambda: tester.test_text_to_speech()[0]),
            ("Steerable TTS", tester.test_steerable_tts),
            ("Transcription", tester.test_transcription_models),
            ("Real-time Performance", tester.test_real_time_performance),
            ("Audio Formats", tester.test_audio_formats)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")
                results[test_name] = False
        
        # Summary
        print(f"\n{'='*50}")
        print("🏁 TEST RESULTS SUMMARY")
        print(f"{'='*50}")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {test_name}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL OPENAI AUDIO TESTS PASSED!")
            print("💡 OpenAI audio models are ready for voice assistant use")
        else:
            print("⚠️  Some tests failed - check API keys and network connection")
        
        # Show generated files
        tester.list_generated_files()
            
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main() 