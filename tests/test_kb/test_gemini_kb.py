#!/usr/bin/env python3
"""
Test Gemini Knowledge Base Functionality

Tests REAL Gemini API calls for:
- Document ingestion and organization
- Information retrieval and search
- Embedding generation and similarity
- Context management

NO MOCKS - All real API calls.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
load_dotenv('config/api_keys.env')

class GeminiKBTester:
    def __init__(self):
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
            
        genai.configure(api_key=self.gemini_key)
        
        # Use latest Gemini 2.5 models
        self.chat_model = genai.GenerativeModel('gemini-2.5-flash')
        self.reasoning_model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Test KB directory
        self.kb_test_dir = Path("kb/test_documents")
        self.kb_test_dir.mkdir(parents=True, exist_ok=True)
        
    def test_document_ingestion(self):
        """Test adding documents to knowledge base."""
        print("🧪 Testing Document Ingestion...")
        
        # Create test documents
        test_docs = {
            "project_info.txt": """
Voice Assistant Project - Jane
==============================

This is a voice-activated personal assistant built with:
- OpenAI audio models for speech-to-text and text-to-speech
- Gemini 2.5 models for intelligent conversation and reasoning
- Local knowledge base for personalized responses
- Wake word activation for hands-free operation

Key features:
- Real-time voice interaction
- Intelligent document organization
- Contextual responses based on personal knowledge
- Multi-language support
""",
            "technical_specs.txt": """
Technical Specifications
=======================

Audio Pipeline:
- Input: 16kHz sampling rate
- Wake word: "Hey Google" or configurable
- Transcription: OpenAI gpt-4o-transcribe
- TTS: OpenAI gpt-4o-mini-tts

AI Models:
- Chat: Gemini 2.5 Flash (thinking model)
- Reasoning: Gemini 2.5 Pro (complex tasks)
- Embeddings: Gemini embedding-exp-03-07

Performance Targets:
- Response time: <5 seconds
- Wake word detection: <500ms
- Transcription accuracy: >95%
""",
            "meeting_notes.txt": """
Meeting Notes - June 17, 2025
=============================

Discussed voice assistant project roadmap:

Phase 1: Audio I/O Foundation
- Set up microphone and speaker testing
- Implement wake word detection
- Test transcription pipeline

Phase 2: Knowledge Base
- Create document ingestion system
- Implement Gemini-based organization
- Test retrieval and search

Phase 3: Response Generation
- Integrate OpenAI TTS
- Add voice personality options
- Test conversation flow

Next meeting: Focus on Phase 1 implementation
Action items: Complete audio hardware testing
"""
        }
        
        # Write test documents
        for filename, content in test_docs.items():
            doc_path = self.kb_test_dir / filename
            with open(doc_path, 'w') as f:
                f.write(content)
            print(f"  ✅ Created: {filename}")
            
        return test_docs
    
    def test_document_analysis(self, test_docs):
        """Test Gemini's ability to analyze and understand documents."""
        print("\n🧠 Testing Document Analysis...")
        
        for filename, content in test_docs.items():
            try:
                prompt = f"""
Analyze this document and provide:
1. Main topic/purpose
2. Key information (3-5 bullet points)
3. Document type classification
4. Suggested tags for organization

Document: {filename}
Content: {content}
"""
                
                response = self.chat_model.generate_content(prompt)
                print(f"\n📄 Analysis of {filename}:")
                print(f"📝 {response.text[:200]}...")
                
            except Exception as e:
                print(f"  ❌ Failed to analyze {filename}: {e}")
                return False
                
        return True
    
    def test_information_retrieval(self, test_docs):
        """Test retrieving specific information from documents."""
        print("\n🔍 Testing Information Retrieval...")
        
        queries = [
            "What audio models are used for transcription?",
            "What are the performance targets for the voice assistant?",
            "When is the next meeting scheduled?",
            "What programming languages or frameworks are mentioned?",
            "What are the main phases of the project?"
        ]
        
        # Combine all documents for context
        full_context = "\n\n".join([f"=== {name} ===\n{content}" 
                                   for name, content in test_docs.items()])
        
        for query in queries:
            try:
                prompt = f"""
Based on the following knowledge base documents, answer this question:
"{query}"

If the information isn't available, say "Not found in knowledge base."
Be specific and cite which document contains the information.

Knowledge Base:
{full_context}
"""
                
                response = self.chat_model.generate_content(prompt)
                print(f"\n❓ Query: {query}")
                print(f"💡 Answer: {response.text}")
                
            except Exception as e:
                print(f"  ❌ Failed query '{query}': {e}")
                return False
                
        return True
    
    def test_complex_reasoning(self):
        """Test Gemini 2.5 Pro's reasoning capabilities."""
        print("\n🧠 Testing Complex Reasoning with Gemini 2.5 Pro...")
        
        reasoning_tasks = [
            {
                "task": "Project Planning Analysis",
                "prompt": """
Based on the voice assistant project information, analyze:
1. What are the potential technical risks in each phase?
2. Which components might have dependencies that could cause delays?
3. What would be the optimal order of implementation?
4. What additional considerations should be planned for?

Think through this step by step.
"""
            },
            {
                "task": "Technical Architecture Review",
                "prompt": """
Review the technical specifications and identify:
1. Any potential bottlenecks in the audio pipeline
2. Scalability concerns with the current model choices
3. Alternative approaches that might improve performance
4. Security considerations for voice data handling

Provide reasoning for each point.
"""
            }
        ]
        
        for task_info in reasoning_tasks:
            try:
                print(f"\n🎯 {task_info['task']}:")
                response = self.reasoning_model.generate_content(task_info['prompt'])
                print(f"🧠 Analysis: {response.text[:300]}...")
                
            except Exception as e:
                print(f"  ❌ Failed reasoning task: {e}")
                return False
                
        return True
    
    def test_embeddings(self):
        """Test embedding generation for semantic search."""
        print("\n🔢 Testing Embedding Generation...")
        
        test_texts = [
            "voice recognition and speech processing",
            "audio transcription using AI models",
            "meeting notes and project planning",
            "technical specifications and performance",
            "knowledge base and document retrieval"
        ]
        
        try:
            embeddings = []
            for text in test_texts:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                embeddings.append(result['embedding'])
                print(f"  ✅ Generated embedding for: '{text}' (dim: {len(result['embedding'])})")
            
            # Test similarity (simple dot product)
            import numpy as np
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append((test_texts[i], test_texts[j], sim))
            
            print(f"\n📊 Similarity Analysis:")
            similarities.sort(key=lambda x: x[2], reverse=True)
            for text1, text2, sim in similarities[:3]:
                print(f"  🔗 '{text1}' ↔ '{text2}': {sim:.3f}")
                
            return True
            
        except Exception as e:
            print(f"  ❌ Embedding test failed: {e}")
            return False
    
    def test_conversation_memory(self):
        """Test maintaining conversation context."""
        print("\n💭 Testing Conversation Memory...")
        
        conversation = [
            "What is the main purpose of the Jane project?",
            "What audio models does it use?",
            "How do those models compare to the Gemini models being used?",
            "What would happen if we switched to use Gemini 2.5 native audio instead?"
        ]
        
        context = ""
        for question in conversation:
            try:
                prompt = f"""
Previous conversation context:
{context}

User question: {question}

Respond naturally, referring to previous context when relevant.
"""
                
                response = self.chat_model.generate_content(prompt)
                print(f"\n👤 User: {question}")
                print(f"🤖 Assistant: {response.text[:150]}...")
                
                # Update context
                context += f"\nUser: {question}\nAssistant: {response.text}\n"
                
            except Exception as e:
                print(f"  ❌ Conversation failed: {e}")
                return False
                
        return True
    
    def cleanup(self):
        """Clean up test files."""
        print("\n🧹 Cleaning up test files...")
        import shutil
        if self.kb_test_dir.exists():
            shutil.rmtree(self.kb_test_dir)
            print("  ✅ Test directory cleaned up")

def main():
    print("🔥 GEMINI KNOWLEDGE BASE TESTING")
    print("=" * 50)
    print("Testing with REAL Gemini 2.5 API calls (NO MOCKS)")
    print("=" * 50)
    
    tester = GeminiKBTester()
    
    try:
        # Run all tests
        test_docs = tester.test_document_ingestion()
        
        tests = [
            ("Document Analysis", lambda: tester.test_document_analysis(test_docs)),
            ("Information Retrieval", lambda: tester.test_information_retrieval(test_docs)),
            ("Complex Reasoning", tester.test_complex_reasoning),
            ("Embeddings", tester.test_embeddings),
            ("Conversation Memory", tester.test_conversation_memory)
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
            print("🎉 ALL GEMINI KB TESTS PASSED!")
            print("💡 Gemini 2.5 models are ready for knowledge base operations")
        else:
            print("⚠️  Some tests failed - check API keys and network connection")
            
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main() 