#!/usr/bin/env python3
"""
Voice Assistant Component Test Runner

Run individual functionality tests without mocks:
- Gemini Knowledge Base operations
- OpenAI Audio generation and transcription
- Hardware microphone and speaker testing
- Full integration testing

Usage:
  python tests/run_tests.py [test_name]
  
Available tests:
  - kb           : Test Gemini knowledge base functionality
  - audio        : Test OpenAI audio models (TTS/STT)
  - hardware     : Test microphone and speaker hardware
  - wakeword     : Test wake word detection engines
  - all          : Run all tests sequentially
  - integration  : Run integration tests (coming soon)

Examples:
  python tests/run_tests.py kb
  python tests/run_tests.py audio
  python tests/run_tests.py hardware
  python tests/run_tests.py wakeword
  python tests/run_tests.py all
"""

import sys
import subprocess
import time
from pathlib import Path

def run_gemini_kb_test():
    """Run Gemini Knowledge Base tests."""
    print("🔥 RUNNING GEMINI KNOWLEDGE BASE TESTS")
    print("=" * 60)
    
    test_path = Path("tests/test_kb/test_gemini_kb.py")
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], cwd=".")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run KB tests: {e}")
        return False

def run_openai_audio_test():
    """Run OpenAI Audio tests."""
    print("🔥 RUNNING OPENAI AUDIO TESTS")
    print("=" * 60)
    
    test_path = Path("tests/test_audio/test_openai_audio.py")
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], cwd=".")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run audio tests: {e}")
        return False

def run_hardware_test():
    """Run Hardware Audio tests."""
    print("🔥 RUNNING HARDWARE AUDIO TESTS")
    print("=" * 60)
    
    test_path = Path("tests/test_audio/test_mic_speaker.py")
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], cwd=".")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run hardware tests: {e}")
        return False

def run_wake_word_test():
    """Run Wake Word Detection tests."""
    print("🔥 RUNNING WAKE WORD DETECTION TESTS")
    print("=" * 60)
    
    test_path = Path("tests/test_audio/test_wake_word.py")
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], cwd=".")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run wake word tests: {e}")
        return False

def run_all_tests():
    """Run all component tests sequentially."""
    print("🔥 RUNNING ALL COMPONENT TESTS")
    print("=" * 60)
    print("This will run all tests sequentially with breaks between them.")
    print("Press Ctrl+C to stop at any point.\n")
    
    tests = [
        ("Gemini Knowledge Base", run_gemini_kb_test),
        ("OpenAI Audio Models", run_openai_audio_test),
        ("Hardware Audio I/O", run_hardware_test),
        ("Wake Word Detection", run_wake_word_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 STARTING: {test_name}")
        print(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n⚠️  Test suite interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
        
        # Brief pause between tests
        if test_name != list(results.keys())[-1]:  # Not the last test
            print(f"\n⏸️  Brief pause before next test...")
            time.sleep(2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 ALL TESTS COMPLETE - SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📊 Overall Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL COMPONENT TESTS PASSED!")
        print("💡 Your voice assistant setup is ready for development")
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed - voice assistant should work with minor issues")
    else:
        print("❌ Multiple component failures - check your setup")
    
    return passed == total

def check_dependencies():
    """Check if test dependencies are installed."""
    print("🔧 Checking Test Dependencies...")
    
    required_packages = [
        ("google-generativeai", "Google Gemini API"),
        ("openai", "OpenAI API"),
        ("sounddevice", "Audio hardware testing"),
        ("numpy", "Audio processing"),
        ("pygame", "Audio playback")
    ]
    
    optional_packages = [
        ("pvporcupine", "Professional wake word detection")
    ]
    
    missing = []
    
    for package, description in required_packages:
        try:
            # Special cases for import names that differ from package names
            if package == "google-generativeai":
                import google.generativeai
            else:
                __import__(package.replace("-", "_"))
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ❌ {package} - {description}")
            missing.append(package)
    
    # Check optional packages
    print(f"\n🔧 Checking Optional Dependencies...")
    for package, description in optional_packages:
        try:
            if package == "pvporcupine":
                import pvporcupine
            else:
                __import__(package.replace("-", "_"))
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ⚠️  {package} - {description} (optional)")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} required packages:")
        print("   Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    else:
        print("✅ All required test dependencies are available")
        return True

def show_help():
    """Show help message."""
    print(__doc__)

def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    test_type = sys.argv[1].lower()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Cannot run tests without required dependencies")
        return
    
    print(f"\n🚀 Starting {test_type} tests...")
    print("⚠️  Remember: NO MOCKS - All tests use REAL APIs and hardware\n")
    
    if test_type == "kb":
        success = run_gemini_kb_test()
    elif test_type == "audio":
        success = run_openai_audio_test()
    elif test_type == "hardware":
        success = run_hardware_test()
    elif test_type == "wakeword":
        success = run_wake_word_test()
    elif test_type == "all":
        success = run_all_tests()
    elif test_type in ["help", "-h", "--help"]:
        show_help()
        return
    else:
        print(f"❌ Unknown test type: {test_type}")
        print("Available tests: kb, audio, hardware, wakeword, all")
        show_help()
        return
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 