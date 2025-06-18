#!/usr/bin/env python3
"""
Test Audio Hardware (Microphone & Speaker)

Tests REAL hardware devices:
- Microphone input detection and recording
- Speaker output and playback
- Audio quality and latency
- Device enumeration and selection

NO MOCKS - All real hardware testing.
"""

import os
import sys
import time
import wave
from pathlib import Path
import threading
import queue

try:
    import sounddevice as sd
    import numpy as np
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  sounddevice not available. Install with: pip install sounddevice")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not available. Install with: pip install pygame")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

class AudioHardwareTester:
    def __init__(self):
        self.sample_rate = 16000  # Voice assistant standard
        self.channels = 1  # Mono for voice
        self.audio_test_dir = Path("tests/test_audio/hardware")
        self.audio_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pygame for audio playback
        if PYGAME_AVAILABLE:
            pygame.mixer.pre_init(frequency=self.sample_rate, size=-16, channels=self.channels)
            pygame.mixer.init()
    
    def check_dependencies(self):
        """Check if required audio libraries are available."""
        print("🔧 Checking Audio Dependencies...")
        
        issues = []
        
        if not SOUNDDEVICE_AVAILABLE:
            issues.append("sounddevice - Required for microphone input")
            
        if not PYGAME_AVAILABLE:
            issues.append("pygame - Required for speaker output")
        
        if issues:
            print("❌ Missing dependencies:")
            for issue in issues:
                print(f"   - {issue}")
            print("\n💡 Install with:")
            print("   pip install sounddevice pygame")
            return False
        else:
            print("✅ All audio dependencies available")
            return True
    
    def list_audio_devices(self):
        """List all available audio input and output devices."""
        if not SOUNDDEVICE_AVAILABLE:
            return False
            
        print("\n🎧 Audio Device Information...")
        
        try:
            devices = sd.query_devices()
            
            print("📥 Input Devices (Microphones):")
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device))
                    default_mark = " (DEFAULT)" if i == sd.default.device[0] else ""
                    print(f"   [{i}] {device['name']}{default_mark}")
                    print(f"       Channels: {device['max_input_channels']}, Rate: {device['default_samplerate']:.0f}Hz")
            
            print("\n📤 Output Devices (Speakers):")
            output_devices = []
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    output_devices.append((i, device))
                    default_mark = " (DEFAULT)" if i == sd.default.device[1] else ""
                    print(f"   [{i}] {device['name']}{default_mark}")
                    print(f"       Channels: {device['max_output_channels']}, Rate: {device['default_samplerate']:.0f}Hz")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to list devices: {e}")
            return False
    
    def test_microphone_levels(self):
        """Test microphone input levels and responsiveness."""
        if not SOUNDDEVICE_AVAILABLE:
            return False
            
        print("\n🎤 Testing Microphone Input...")
        print("   Speak into your microphone for 5 seconds...")
        
        try:
            # Record audio for level testing
            duration = 5  # seconds
            print(f"   Recording for {duration} seconds... Speak now!")
            
            recording = sd.rec(int(duration * self.sample_rate), 
                             samplerate=self.sample_rate, 
                             channels=self.channels,
                             dtype='float64')
            
            # Real-time level monitoring
            start_time = time.time()
            while time.time() - start_time < duration:
                # Get current frame for level display
                current_time = time.time() - start_time
                progress = int((current_time / duration) * 20)
                bar = "█" * progress + "░" * (20 - progress)
                print(f"\r   [{bar}] {current_time:.1f}s", end="", flush=True)
                time.sleep(0.1)
            
            sd.wait()  # Wait for recording to complete
            print("\n   ✅ Recording complete!")
            
            # Analyze recording
            max_level = np.max(np.abs(recording))
            rms_level = np.sqrt(np.mean(recording**2))
            
            print(f"\n📊 Audio Analysis:")
            print(f"   Max level: {max_level:.3f} ({20*np.log10(max_level + 1e-10):.1f} dB)")
            print(f"   RMS level: {rms_level:.3f} ({20*np.log10(rms_level + 1e-10):.1f} dB)")
            
            # Save recording for playback test
            recording_path = self.audio_test_dir / "mic_test.wav"
            self.save_wav(recording_path, recording, self.sample_rate)
            print(f"   💾 Saved recording: {recording_path}")
            
            # Quality assessment
            if max_level < 0.001:
                print("   ⚠️  Very low input level - check microphone connection")
                return False
            elif max_level > 0.9:
                print("   ⚠️  Input level too high - may cause clipping")
                return False
            else:
                print("   ✅ Good input levels detected")
                return True
                
        except Exception as e:
            print(f"\n❌ Microphone test failed: {e}")
            return False
    
    def test_speaker_output(self):
        """Test speaker output with different frequencies and volumes."""
        if not SOUNDDEVICE_AVAILABLE:
            return False
            
        print("\n🔊 Testing Speaker Output...")
        
        try:
            # Test with different frequencies
            test_frequencies = [440, 880, 1000]  # A4, A5, 1kHz
            duration = 1.0  # seconds per tone
            
            for freq in test_frequencies:
                print(f"   🎵 Playing {freq}Hz tone...")
                
                # Generate sine wave
                t = np.linspace(0, duration, int(self.sample_rate * duration), False)
                tone = 0.3 * np.sin(2 * np.pi * freq * t)  # 30% volume
                
                # Play tone
                sd.play(tone, self.sample_rate)
                sd.wait()
                
                print(f"   ✅ {freq}Hz tone completed")
                time.sleep(0.5)  # Brief pause between tones
            
            # Test with recorded audio if available
            mic_recording = self.audio_test_dir / "mic_test.wav"
            if mic_recording.exists():
                print("\n   🎤➡️🔊 Playing back microphone recording...")
                recording, sample_rate = self.load_wav(mic_recording)
                
                if recording is not None:
                    sd.play(recording, sample_rate)
                    sd.wait()
                    print("   ✅ Playback completed")
                else:
                    print("   ❌ Could not load recording")
            
            return True
            
        except Exception as e:
            print(f"❌ Speaker test failed: {e}")
            return False
    
    def test_audio_latency(self):
        """Test audio input/output latency."""
        if not SOUNDDEVICE_AVAILABLE:
            return False
            
        print("\n⚡ Testing Audio Latency...")
        print("   This test measures round-trip audio latency")
        print("   Make sure speakers and microphone can hear each other")
        
        try:
            # Generate a click sound
            click_duration = 0.1  # 100ms
            t = np.linspace(0, click_duration, int(self.sample_rate * click_duration), False)
            click = 0.5 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 20)  # Decaying sine
            
            # Record while playing (to measure latency)
            record_duration = 2.0  # seconds
            print("   🔄 Measuring latency... (you may hear a click)")
            
            # Start recording
            recording = sd.playrec(click, 
                                 samplerate=self.sample_rate, 
                                 channels=self.channels,
                                 output_mapping=[1],  # output to channel 1
                                 input_mapping=[1])   # input from channel 1
            sd.wait()
            
            # Find the latency by detecting the echo
            recording_flat = recording.flatten() if recording.ndim > 1 else recording
            
            # Find peaks to estimate latency
            threshold = 0.1 * np.max(np.abs(recording_flat))
            peaks = np.where(np.abs(recording_flat) > threshold)[0]
            
            if len(peaks) > 0:
                latency_samples = peaks[0]
                latency_ms = (latency_samples / self.sample_rate) * 1000
                print(f"   📊 Estimated latency: {latency_ms:.1f}ms")
                
                if latency_ms < 50:
                    print("   ✅ Excellent latency for real-time use")
                elif latency_ms < 100:
                    print("   ✅ Good latency for voice assistant")
                else:
                    print("   ⚠️  High latency - may affect user experience")
                
                return True
            else:
                print("   ⚠️  Could not detect audio return - check speaker/mic setup")
                return False
                
        except Exception as e:
            print(f"❌ Latency test failed: {e}")
            return False
    
    def test_noise_floor(self):
        """Test background noise levels."""
        if not SOUNDDEVICE_AVAILABLE:
            return False
            
        print("\n🤫 Testing Background Noise...")
        print("   Stay quiet for 3 seconds to measure noise floor...")
        
        try:
            duration = 3  # seconds
            print("   🔇 Recording silence...")
            
            recording = sd.rec(int(duration * self.sample_rate), 
                             samplerate=self.sample_rate, 
                             channels=self.channels,
                             dtype='float64')
            sd.wait()
            
            # Analyze noise floor
            rms_noise = np.sqrt(np.mean(recording**2))
            noise_db = 20 * np.log10(rms_noise + 1e-10)
            
            print(f"   📊 Noise floor: {rms_noise:.6f} ({noise_db:.1f} dB)")
            
            if noise_db < -40:
                print("   ✅ Excellent - Very quiet environment")
            elif noise_db < -30:
                print("   ✅ Good - Suitable for voice assistant")
            elif noise_db < -20:
                print("   ⚠️  Moderate noise - may affect wake word detection")
            else:
                print("   ⚠️  High noise level - consider noise reduction")
            
            return True
            
        except Exception as e:
            print(f"❌ Noise floor test failed: {e}")
            return False
    
    def test_simultaneous_audio(self):
        """Test simultaneous recording and playback."""
        if not SOUNDDEVICE_AVAILABLE:
            return False
            
        print("\n🔄 Testing Simultaneous Record/Playback...")
        print("   This simulates voice assistant operation")
        
        try:
            # Create a test tone to play while recording
            tone_duration = 3.0
            freq = 600  # Hz
            t = np.linspace(0, tone_duration, int(self.sample_rate * tone_duration), False)
            test_tone = 0.2 * np.sin(2 * np.pi * freq * t)
            
            print("   🎵 Playing tone while recording...")
            
            # Simultaneous playback and recording
            recording = sd.playrec(test_tone, 
                                 samplerate=self.sample_rate, 
                                 channels=self.channels,
                                 dtype='float64')
            sd.wait()
            
            # Analyze the recording
            max_level = np.max(np.abs(recording))
            
            # Save simultaneous test
            simul_path = self.audio_test_dir / "simultaneous_test.wav"
            self.save_wav(simul_path, recording, self.sample_rate)
            
            print(f"   📊 Recorded level: {max_level:.3f}")
            print(f"   💾 Saved: {simul_path}")
            
            if max_level > 0.01:
                print("   ✅ Simultaneous audio operation successful")
                return True
            else:
                print("   ⚠️  Low recorded level during simultaneous operation")
                return False
                
        except Exception as e:
            print(f"❌ Simultaneous audio test failed: {e}")
            return False
    
    def save_wav(self, filepath, data, sample_rate):
        """Save audio data to WAV file."""
        try:
            # Ensure data is in the right format
            if data.dtype != np.int16:
                # Convert float to int16
                data_int16 = np.int16(data * 32767)
            else:
                data_int16 = data
            
            with wave.open(str(filepath), 'w') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(data_int16.tobytes())
            
            return True
        except Exception as e:
            print(f"   ❌ Failed to save WAV: {e}")
            return False
    
    def load_wav(self, filepath):
        """Load audio data from WAV file."""
        try:
            with wave.open(str(filepath), 'r') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to float
                audio_float = audio_data.astype(np.float64) / 32767.0
                
                return audio_float, sample_rate
                
        except Exception as e:
            print(f"   ❌ Failed to load WAV: {e}")
            return None, None
    
    def list_generated_files(self):
        """List generated audio files for manual inspection."""
        print("\n📁 Generated Hardware Test Files:")
        
        test_files = list(self.audio_test_dir.glob("*.wav"))
        
        if not test_files:
            print("   No test files generated")
            return
        
        for test_file in sorted(test_files):
            file_size = os.path.getsize(test_file)
            print(f"   🎵 {test_file.name} ({file_size:,} bytes)")
        
        print(f"\n💡 To manually test files:")
        print(f"   cd {self.audio_test_dir}")
        print(f"   open *.wav  # (on macOS)")
    
    def cleanup(self):
        """Cleanup resources."""
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        
        print(f"\n🧹 Hardware test files preserved in: {self.audio_test_dir}")

def main():
    print("🔥 AUDIO HARDWARE TESTING")
    print("=" * 50)
    print("Testing with REAL microphone and speakers (NO MOCKS)")
    print("=" * 50)
    
    tester = AudioHardwareTester()
    
    # Check dependencies first
    if not tester.check_dependencies():
        print("\n❌ Cannot proceed without required audio libraries")
        return
    
    try:
        # Run all tests
        tests = [
            ("Device Enumeration", tester.list_audio_devices),
            ("Microphone Input", tester.test_microphone_levels),
            ("Speaker Output", tester.test_speaker_output),
            ("Background Noise", tester.test_noise_floor),
            ("Audio Latency", tester.test_audio_latency),
            ("Simultaneous I/O", tester.test_simultaneous_audio)
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
        print("🏁 HARDWARE TEST RESULTS SUMMARY")
        print(f"{'='*50}")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {test_name}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL HARDWARE TESTS PASSED!")
            print("💡 Your microphone and speakers are ready for voice assistant use")
        elif passed >= total * 0.7:
            print("⚠️  Most tests passed - voice assistant should work with minor issues")
        else:
            print("❌ Multiple hardware issues detected - check your audio setup")
        
        # Show generated files
        tester.list_generated_files()
        
        # Final recommendations
        print(f"\n💡 Recommendations:")
        if results.get("Microphone Input", False):
            print("   ✅ Microphone is working well")
        else:
            print("   ⚠️  Check microphone connection and permissions")
            
        if results.get("Speaker Output", False):
            print("   ✅ Speakers are working well")
        else:
            print("   ⚠️  Check speaker connection and volume")
            
        if results.get("Audio Latency", False):
            print("   ✅ Audio latency is acceptable")
        else:
            print("   ⚠️  High latency may affect real-time performance")
            
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main() 