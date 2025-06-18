#!/usr/bin/env python3
"""
API Testing Script for Voice Assistant

Tests real API access to:
- OpenAI (GPT, Audio models)
- Google Gemini (Chat, Embedding models) 
- Perplexity (Chat models)

NO MOCK DATA - All real API calls only.
"""

import os
import asyncio
from dotenv import load_dotenv
import openai
import google.generativeai as genai
import requests
import json

# Load environment variables
load_dotenv('config/api_keys.env')

class APITester:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        
        # Initialize clients
        self.openai_client = openai.OpenAI(api_key=self.openai_key)
        genai.configure(api_key=self.gemini_key)
        
    def test_openai_api(self):
        """Test OpenAI API access and list available models"""
        print("🔥 TESTING OPENAI API")
        print("="*50)
        
        try:
            # List all available models
            models = self.openai_client.models.list()
            print(f"✅ OpenAI API Access: SUCCESS")
            print(f"📊 Total Models Available: {len(models.data)}")
            
            # Categorize models
            gpt_models = []
            audio_models = []
            other_models = []
            
            for model in models.data:
                model_id = model.id
                if 'gpt' in model_id.lower():
                    gpt_models.append(model_id)
                elif any(audio_term in model_id.lower() for audio_term in ['whisper', 'tts', 'audio', 'transcribe']):
                    audio_models.append(model_id)
                else:
                    other_models.append(model_id)
            
            print(f"\n🤖 GPT Models ({len(gpt_models)}):")
            for model in sorted(gpt_models):
                print(f"  - {model}")
            
            print(f"\n🎤 Audio Models ({len(audio_models)}):")
            for model in sorted(audio_models):
                print(f"  - {model}")
                
            print(f"\n🔧 Other Models ({len(other_models)}):")
            for model in sorted(other_models)[:10]:  # Show first 10
                print(f"  - {model}")
            if len(other_models) > 10:
                print(f"  ... and {len(other_models) - 10} more")
            
            # Test specific audio models we need
            print(f"\n🎯 CHECKING REQUIRED AUDIO MODELS:")
            required_models = [
                'gpt-4o-transcribe',
                'gpt-4o-mini-transcribe', 
                'gpt-4o-mini-tts',
                'whisper-1'
            ]
            
            for model in required_models:
                if model in [m.id for m in models.data]:
                    print(f"  ✅ {model} - AVAILABLE")
                else:
                    print(f"  ❌ {model} - NOT FOUND")
            
            # Test a simple chat completion
            print(f"\n🧪 Testing Chat Completion...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'OpenAI API test successful'"}],
                max_tokens=20
            )
            print(f"✅ Chat Test: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
        
        print("\n")
    
    def test_gemini_api(self):
        """Test Gemini API access and list available models"""
        print("💎 TESTING GEMINI API")
        print("="*50)
        
        try:
            # List available models
            models = genai.list_models()
            model_list = list(models)
            
            print(f"✅ Gemini API Access: SUCCESS")
            print(f"📊 Total Models Available: {len(model_list)}")
            
            # Categorize models
            chat_models = []
            embedding_models = []
            other_models = []
            
            for model in model_list:
                model_name = model.name
                if 'embed' in model_name.lower():
                    embedding_models.append(model_name)
                elif any(term in model_name.lower() for term in ['gemini', 'pro', 'flash']):
                    chat_models.append(model_name)
                else:
                    other_models.append(model_name)
            
            print(f"\n🤖 Chat Models ({len(chat_models)}):")
            for model in sorted(chat_models):
                print(f"  - {model}")
            
            print(f"\n🔗 Embedding Models ({len(embedding_models)}):")
            for model in sorted(embedding_models):
                print(f"  - {model}")
                
            print(f"\n🔧 Other Models ({len(other_models)}):")
            for model in sorted(other_models):
                print(f"  - {model}")
            
            # Test specific models we need
            print(f"\n🎯 CHECKING REQUIRED MODELS:")
            required_models = [
                'models/gemini-1.5-flash',
                'models/gemini-embedding-exp-03-07',
                'models/text-embedding-004'
            ]
            
            print(f"\n🔥 CHECKING LATEST GEMINI 2.5 MODELS:")
            gemini_25_models = [
                'models/gemini-2.5-flash',
                'models/gemini-2.5-pro', 
                'models/gemini-2.5-flash-lite-preview-06-17',
                'models/gemini-2.5-flash-preview-tts'
            ]
            
            available_names = [m.name for m in model_list]
            
            for model in required_models:
                if model in available_names:
                    print(f"  ✅ {model} - AVAILABLE")
                else:
                    print(f"  ❌ {model} - NOT FOUND")
                    
            for model in gemini_25_models:
                if model in available_names:
                    print(f"  🔥 {model} - AVAILABLE")
                else:
                    print(f"  ❌ {model} - NOT FOUND")
            
            # Test chat generation with Gemini 2.5 Flash
            print(f"\n🧪 Testing Gemini 2.5 Flash Chat Generation...")
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content("Say 'Gemini 2.5 Flash working perfectly!'")
                print(f"✅ Gemini 2.5 Flash Test: {response.text}")
            except Exception as e:
                print(f"⚠️ Gemini 2.5 Flash Failed: {e}")
                # Fallback to 1.5 Flash
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content("Say 'Gemini 1.5 Flash fallback working'")
                print(f"✅ Fallback 1.5 Flash Test: {response.text}")
                
            # Test Gemini 2.5 Pro if available
            print(f"\n🧪 Testing Gemini 2.5 Pro (Most Intelligent)...")
            try:
                model = genai.GenerativeModel('gemini-2.5-pro')
                response = model.generate_content("Say 'Gemini 2.5 Pro - thinking model ready!'")
                print(f"🧠 Gemini 2.5 Pro Test: {response.text}")
            except Exception as e:
                print(f"⚠️ Gemini 2.5 Pro Failed: {e}")
            
            # Test embedding generation if available
            print(f"\n🧪 Testing Embedding Generation...")
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content="Test embedding generation"
                )
                print(f"✅ Embedding Test: Generated {len(result['embedding'])} dimensions")
            except Exception as e:
                print(f"⚠️ Embedding Test Failed: {e}")
                
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
        
        print("\n")
    
    def test_perplexity_api(self):
        """Test Perplexity API access and list available models"""
        print("🌊 TESTING PERPLEXITY API")
        print("="*50)
        
        try:
            # Perplexity uses OpenAI-compatible API
            headers = {
                "Authorization": f"Bearer {self.perplexity_key}",
                "Content-Type": "application/json"
            }
            
            # Test available models (if endpoint exists)
            print("🧪 Testing Perplexity Chat Completion...")
            
            # Test chat completion
            data = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {"role": "user", "content": "Say 'Perplexity API test successful'"}
                ],
                "max_tokens": 20
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Perplexity API Access: SUCCESS")
                print(f"✅ Chat Test: {result['choices'][0]['message']['content']}")
                
                # Common Perplexity models (they don't have a list endpoint)
                print(f"\n🤖 Known Perplexity Models:")
                known_models = [
                    "llama-3.1-sonar-small-128k-online",
                    "llama-3.1-sonar-large-128k-online", 
                    "llama-3.1-sonar-huge-128k-online",
                    "llama-3.1-8b-instruct",
                    "llama-3.1-70b-instruct",
                    "mixtral-8x7b-instruct"
                ]
                
                for model in known_models:
                    print(f"  - {model}")
                    
            else:
                print(f"❌ Perplexity API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Perplexity API Error: {e}")
        
        print("\n")
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 VOICE ASSISTANT API TESTING")
        print("="*60)
        print("Testing ALL APIs with REAL endpoints (NO MOCKS)")
        print("="*60)
        print()
        
        self.test_openai_api()
        self.test_gemini_api()
        self.test_perplexity_api()
        
        print("🏁 API TESTING COMPLETE")
        print("="*60)

def main():
    """Main function to run API tests"""
    tester = APITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 