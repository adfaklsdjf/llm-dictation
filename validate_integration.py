#!/usr/bin/env python3
"""
Quick integration validator for LLM Dictation.

This script validates that all components can work together without external dependencies.
It catches common integration issues like missing methods or incompatible interfaces.
"""

import asyncio
import inspect
import sys
from pathlib import Path
from typing import List, Any
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from src.audio.recorder import AudioRecorder
        from src.audio.transcriber import WhisperTranscriber, TranscriptionResult
        from src.cleanup.providers import CleanupProvider, RuleBasedProvider, CleanupResult
        from src.cleanup.cleaner import TextCleaner
        from src.ui.terminal import TerminalUI
        from src.main import DictationApp
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_component_initialization():
    """Test that all components can be initialized."""
    print("🔍 Testing component initialization...")
    
    try:
        from src.audio.recorder import AudioRecorder
        from src.audio.transcriber import WhisperTranscriber
        from src.cleanup.cleaner import TextCleaner
        from src.ui.terminal import TerminalUI
        from src.main import DictationApp
        
        recorder = AudioRecorder()
        transcriber = WhisperTranscriber()
        cleaner = TextCleaner()
        ui = TerminalUI()
        app = DictationApp()
        
        print("✅ All components initialize successfully")
        return True
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_existence():
    """Test that all required methods exist on components."""
    print("🔍 Testing method existence...")
    
    try:
        from src.main import DictationApp
        
        app = DictationApp()
        
        # Check AudioRecorder methods
        required_recorder_methods = ['start_recording', 'stop_recording', 'is_recording']
        for method in required_recorder_methods:
            if not hasattr(app.recorder, method):
                raise AttributeError(f"AudioRecorder missing method: {method}")
        
        # Check WhisperTranscriber methods
        required_transcriber_methods = ['transcribe', 'get_available_models']
        for method in required_transcriber_methods:
            if not hasattr(app.transcriber, method):
                raise AttributeError(f"WhisperTranscriber missing method: {method}")
        
        # Check TextCleaner methods
        required_cleaner_methods = ['cleanup_text', 'get_available_providers']
        for method in required_cleaner_methods:
            if not hasattr(app.cleaner, method):
                raise AttributeError(f"TextCleaner missing method: {method}")
        
        # Check TerminalUI methods (these were causing the original error)
        required_ui_methods = [
            'prompt_start_recording', 'show_recording_status', 'prompt_stop_recording',
            'show_transcription_progress', 'display_cleanup_results', 
            'show_error', 'show_success'
        ]
        for method in required_ui_methods:
            if not hasattr(app.ui, method):
                raise AttributeError(f"TerminalUI missing method: {method}")
        
        # Verify that show_welcome is NOT called (this was the bug)
        # The welcome functionality should be in prompt_start_recording
        if hasattr(app.ui, 'show_welcome'):
            print("⚠️  Warning: TerminalUI has show_welcome method but it's not used in main.py")
        
        print("✅ All required methods exist")
        return True
    except Exception as e:
        print(f"❌ Method existence check failed: {e}")
        return False

def test_async_methods():
    """Test that async methods are properly marked as async."""
    print("🔍 Testing async method signatures...")
    
    try:
        from src.main import DictationApp
        
        app = DictationApp()
        
        # Check that expected methods are async
        async_methods = [
            (app.recorder, 'start_recording'),
            (app.recorder, 'stop_recording'),
            (app.transcriber, 'transcribe'),
            (app.cleaner, 'cleanup_text'),
            (app.ui, 'prompt_start_recording'),
            (app.ui, 'show_recording_status'),
            (app.ui, 'prompt_stop_recording'),
            (app.ui, 'show_transcription_progress'),
            (app.ui, 'display_cleanup_results'),
            (app.ui, 'show_error'),
            (app.ui, 'show_success'),
        ]
        
        for obj, method_name in async_methods:
            method = getattr(obj, method_name)
            if not inspect.iscoroutinefunction(method):
                raise TypeError(f"{obj.__class__.__name__}.{method_name} should be async")
        
        print("✅ All async method signatures correct")
        return True
    except Exception as e:
        print(f"❌ Async method check failed: {e}")
        return False

async def test_rule_based_cleanup():
    """Test that rule-based cleanup works without external dependencies."""
    print("🔍 Testing rule-based text cleanup...")
    
    try:
        from src.cleanup.providers import RuleBasedProvider
        
        provider = RuleBasedProvider()
        
        if not provider.is_available():
            raise RuntimeError("RuleBasedProvider should always be available")
        
        test_text = "Um, so I was like, thinking about, uh, creating this application."
        result = await provider.cleanup_text(test_text)
        
        if not result:
            raise RuntimeError("RuleBasedProvider returned no result")
        
        if result.error:
            raise RuntimeError(f"RuleBasedProvider returned error: {result.error}")
        
        # Check that filler words were removed
        cleaned = result.cleaned_text.lower()
        if "um" in cleaned or "uh" in cleaned:
            print(f"⚠️  Warning: Filler words still present in: {result.cleaned_text}")
        
        print(f"✅ Rule-based cleanup works")
        print(f"   Original: {test_text}")
        print(f"   Cleaned:  {result.cleaned_text}")
        return True
    except Exception as e:
        print(f"❌ Rule-based cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mock_integration():
    """Test the full pipeline with mocked user interaction."""
    print("🔍 Testing mocked integration pipeline...")
    
    try:
        from src.main import DictationApp
        from src.cleanup.providers import CleanupResult
        from src.audio.transcriber import TranscriptionResult
        
        app = DictationApp()
        
        # Mock the components to avoid actual audio/API calls
        app.recorder.start_recording = AsyncMock()
        app.recorder.stop_recording = AsyncMock(return_value=b"fake audio")
        app.recorder.is_recording = Mock(return_value=False)
        
        app.transcriber.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="um, this is a test message",
            language="en",
            confidence=0.95
        ))
        
        app.cleaner.cleanup_text = AsyncMock(return_value=[
            CleanupResult(
                original_text="um, this is a test message",
                cleaned_text="This is a test message.",
                provider="test",
                processing_time=1.0,
                quality_score=8.5
            )
        ])
        
        # Mock UI to avoid interactive prompts
        app.ui.prompt_start_recording = AsyncMock(return_value=True)
        app.ui.show_recording_status = AsyncMock()
        app.ui.prompt_stop_recording = AsyncMock(return_value=True) 
        app.ui.show_transcription_progress = AsyncMock()
        app.ui.display_cleanup_results = AsyncMock(return_value=0)  # Select first result
        app.ui.show_success = AsyncMock()
        app.ui.show_error = AsyncMock()
        
        # Mock clipboard to avoid actual clipboard operations
        import unittest.mock
        with unittest.mock.patch('pyperclip.copy'):
            # This should complete without errors
            await app.run_session()
        
        print("✅ Mocked integration pipeline completed successfully")
        return True
    except Exception as e:
        print(f"❌ Mocked integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all validation tests."""
    print("🧪 LLM Dictation Integration Validator")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Component Initialization", test_component_initialization), 
        ("Method Existence", test_method_existence),
        ("Async Method Signatures", test_async_methods),
        ("Rule-Based Cleanup", test_rule_based_cleanup),
        ("Mock Integration", test_mock_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if inspect.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\nThe application should work correctly. You can now run:")
        print("   python3 -m src.main")
        return True
    else:
        print("💥 Some integration tests failed!")
        print("   Fix the issues above before running the application.")
        return False

if __name__ == "__main__":
    asyncio.run(main())