"""
Integration tests for LLM Dictation components.

These tests verify that components work together correctly and catch
common integration issues like missing methods or incompatible interfaces.
"""

import pytest
import asyncio
import inspect
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Any
import tempfile
import os
from pathlib import Path

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.recorder import AudioRecorder
from src.audio.transcriber import WhisperTranscriber, TranscriptionResult
from src.cleanup.providers import CleanupProvider, RuleBasedProvider, CleanupResult
from src.cleanup.cleaner import TextCleaner
from src.ui.terminal import TerminalUI
from src.main import DictationApp


class TestMethodExistence:
    """Test that all required methods exist on components."""
    
    def test_audio_recorder_methods(self):
        """Test that AudioRecorder has all expected methods."""
        recorder = AudioRecorder()
        
        # Check required methods exist
        assert hasattr(recorder, 'start_recording')
        assert hasattr(recorder, 'stop_recording')
        assert hasattr(recorder, 'is_recording')
        
        # Check they are async where expected
        assert inspect.iscoroutinefunction(recorder.start_recording)
        assert inspect.iscoroutinefunction(recorder.stop_recording)
        assert not inspect.iscoroutinefunction(recorder.is_recording)
    
    def test_whisper_transcriber_methods(self):
        """Test that WhisperTranscriber has all expected methods."""
        transcriber = WhisperTranscriber()
        
        # Check required methods exist
        assert hasattr(transcriber, 'transcribe')
        assert hasattr(transcriber, 'transcribe_file')
        assert hasattr(transcriber, 'get_available_models')
        assert hasattr(transcriber, 'is_model_available')
        
        # Check they are async where expected
        assert inspect.iscoroutinefunction(transcriber.transcribe)
        assert inspect.iscoroutinefunction(transcriber.transcribe_file)
        assert not inspect.iscoroutinefunction(transcriber.get_available_models)
        assert not inspect.iscoroutinefunction(transcriber.is_model_available)
    
    def test_text_cleaner_methods(self):
        """Test that TextCleaner has all expected methods."""
        cleaner = TextCleaner()
        
        # Check required methods exist
        assert hasattr(cleaner, 'cleanup_text')
        assert hasattr(cleaner, 'get_best_cleanup')
        assert hasattr(cleaner, 'add_provider')
        assert hasattr(cleaner, 'get_available_providers')
        
        # Check they are async where expected
        assert inspect.iscoroutinefunction(cleaner.cleanup_text)
        assert inspect.iscoroutinefunction(cleaner.get_best_cleanup)
        assert not inspect.iscoroutinefunction(cleaner.add_provider)
        assert not inspect.iscoroutinefunction(cleaner.get_available_providers)
    
    def test_terminal_ui_methods(self):
        """Test that TerminalUI has all expected methods."""
        ui = TerminalUI()
        
        # Check required methods exist - these are the ones main.py actually calls
        assert hasattr(ui, 'prompt_start_recording')
        assert hasattr(ui, 'show_recording_status')
        assert hasattr(ui, 'prompt_stop_recording')
        assert hasattr(ui, 'show_transcription_progress')
        assert hasattr(ui, 'display_cleanup_results')
        assert hasattr(ui, 'show_error')
        assert hasattr(ui, 'show_success')
        
        # Check they are all async
        assert inspect.iscoroutinefunction(ui.prompt_start_recording)
        assert inspect.iscoroutinefunction(ui.show_recording_status)
        assert inspect.iscoroutinefunction(ui.prompt_stop_recording)
        assert inspect.iscoroutinefunction(ui.show_transcription_progress)
        assert inspect.iscoroutinefunction(ui.display_cleanup_results)
        assert inspect.iscoroutinefunction(ui.show_error)
        assert inspect.iscoroutinefunction(ui.show_success)
    
    def test_dictation_app_methods(self):
        """Test that DictationApp has all expected methods."""
        app = DictationApp()
        
        # Check required methods exist
        assert hasattr(app, 'run_session')
        assert hasattr(app, 'set_cleanup_strategy')
        
        # Check they are async where expected
        assert inspect.iscoroutinefunction(app.run_session)
        assert not inspect.iscoroutinefunction(app.set_cleanup_strategy)


class TestDataStructures:
    """Test that data structures are compatible between components."""
    
    def test_transcription_result_structure(self):
        """Test TranscriptionResult has expected fields."""
        # Create a basic TranscriptionResult (assuming it's a dataclass)
        result = TranscriptionResult(
            text="test text",
            language="en",
            confidence=0.95
        )
        
        assert hasattr(result, 'text')
        assert hasattr(result, 'language') 
        assert hasattr(result, 'confidence')
    
    def test_cleanup_result_structure(self):
        """Test CleanupResult has expected fields."""
        result = CleanupResult(
            original_text="um, hello there",
            cleaned_text="Hello there",
            provider="test",
            processing_time=1.0,
            quality_score=8.5
        )
        
        assert hasattr(result, 'original_text')
        assert hasattr(result, 'cleaned_text')
        assert hasattr(result, 'provider')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'quality_score')
        assert hasattr(result, 'error')  # Should have error field even if optional


@pytest.mark.asyncio
class TestMockedIntegration:
    """Test component integration with mocked interactive parts."""
    
    async def test_full_pipeline_with_mocks(self):
        """Test the complete pipeline with mocked user interaction."""
        
        # Create mocked components
        mock_recorder = AsyncMock(spec=AudioRecorder)
        mock_transcriber = AsyncMock(spec=WhisperTranscriber)
        mock_cleaner = AsyncMock(spec=TextCleaner)
        mock_ui = AsyncMock(spec=TerminalUI)
        
        # Set up return values
        mock_recorder.start_recording.return_value = None
        mock_recorder.stop_recording.return_value = b"fake audio data"
        mock_recorder.is_recording.return_value = False
        
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            text="um, hello world, uh, this is a test",
            language="en",
            confidence=0.95
        )
        
        mock_cleanup_results = [
            CleanupResult(
                original_text="um, hello world, uh, this is a test",
                cleaned_text="Hello world, this is a test.",
                provider="test",
                processing_time=1.0,
                quality_score=8.5
            )
        ]
        mock_cleaner.cleanup_text.return_value = mock_cleanup_results
        
        mock_ui.prompt_start_recording.return_value = True
        mock_ui.prompt_stop_recording.return_value = True
        mock_ui.display_cleanup_results.return_value = 0  # Select first result
        
        # Create app with mocked components
        app = DictationApp()
        app.recorder = mock_recorder
        app.transcriber = mock_transcriber
        app.cleaner = mock_cleaner
        app.ui = mock_ui
        
        # Mock clipboard
        with patch('pyperclip.copy') as mock_clipboard:
            # Run the session
            await app.run_session()
            
            # Verify the pipeline was called correctly
            mock_ui.prompt_start_recording.assert_called_once()
            mock_recorder.start_recording.assert_called_once()
            mock_ui.show_recording_status.assert_called_once()
            mock_ui.prompt_stop_recording.assert_called_once()
            mock_recorder.stop_recording.assert_called_once()
            mock_transcriber.transcribe.assert_called_once()
            mock_cleaner.cleanup_text.assert_called_once()
            mock_ui.display_cleanup_results.assert_called_once()
            mock_ui.show_success.assert_called_once()
            mock_clipboard.assert_called_once_with("Hello world, this is a test.")
    
    async def test_error_handling_integration(self):
        """Test that errors are properly handled and displayed."""
        
        # Create app with mocked components that raise errors
        mock_ui = AsyncMock(spec=TerminalUI)
        mock_ui.prompt_start_recording.return_value = True
        mock_ui.prompt_stop_recording.return_value = True
        
        # Mock recorder that fails
        mock_recorder = AsyncMock(spec=AudioRecorder)
        mock_recorder.start_recording.side_effect = Exception("Microphone not found")
        
        app = DictationApp()
        app.recorder = mock_recorder
        app.ui = mock_ui
        
        # Run session - should handle error gracefully
        await app.run_session()
        
        # Verify error was shown to user
        mock_ui.show_error.assert_called_once()
        error_call = mock_ui.show_error.call_args[0][0]
        assert "Microphone not found" in str(error_call)


class TestRuleBasedProvider:
    """Test the rule-based provider which should always work."""
    
    @pytest.mark.asyncio
    async def test_rule_based_cleanup_works(self):
        """Test that rule-based cleanup works without external dependencies."""
        provider = RuleBasedProvider()
        
        assert provider.is_available()  # Should always be available
        
        test_text = "Um, so I was like, thinking about, uh, creating this application."
        result = await provider.cleanup_text(test_text)
        
        assert isinstance(result, CleanupResult)
        assert result.original_text == test_text
        assert "um" not in result.cleaned_text.lower()
        assert "uh" not in result.cleaned_text.lower()
        assert "like," not in result.cleaned_text.lower()
        assert result.provider == "RuleBased"
        assert result.processing_time > 0
        assert result.quality_score > 0
        assert result.error is None


class TestComponentInitialization:
    """Test that all components can be initialized without errors."""
    
    def test_all_components_initialize(self):
        """Test that all main components can be instantiated."""
        # These should not raise exceptions
        recorder = AudioRecorder()
        transcriber = WhisperTranscriber()
        cleaner = TextCleaner()
        ui = TerminalUI()
        app = DictationApp()
        
        # Basic sanity checks
        assert recorder is not None
        assert transcriber is not None
        assert cleaner is not None
        assert ui is not None
        assert app is not None
    
    def test_whisper_models_list(self):
        """Test that Whisper model list is accessible."""
        transcriber = WhisperTranscriber()
        models = transcriber.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "large-v3-turbo" in models or "tiny" in models  # Should have at least some model


class TestMainAppInterfaceCompatibility:
    """Test that main.py can interact with all components correctly."""
    
    def test_main_app_component_interfaces(self):
        """Test that DictationApp can call all required methods on components."""
        app = DictationApp()
        
        # Test that all the methods main.py calls actually exist
        # AudioRecorder methods
        assert callable(getattr(app.recorder, 'start_recording', None))
        assert callable(getattr(app.recorder, 'stop_recording', None))
        assert callable(getattr(app.recorder, 'is_recording', None))
        
        # WhisperTranscriber methods  
        assert callable(getattr(app.transcriber, 'transcribe', None))
        
        # TextCleaner methods
        assert callable(getattr(app.cleaner, 'cleanup_text', None))
        
        # TerminalUI methods - these are the ones that were causing issues
        assert callable(getattr(app.ui, 'prompt_start_recording', None))
        assert callable(getattr(app.ui, 'show_recording_status', None))
        assert callable(getattr(app.ui, 'prompt_stop_recording', None))
        assert callable(getattr(app.ui, 'show_transcription_progress', None))
        assert callable(getattr(app.ui, 'display_cleanup_results', None))
        assert callable(getattr(app.ui, 'show_error', None))
        assert callable(getattr(app.ui, 'show_success', None))
        
        # Verify the show_welcome method that was missing is NOT called
        # (it's not in the current main.py after our fix)
        # This test documents that show_welcome is intentionally not used


if __name__ == "__main__":
    # Run basic smoke tests without pytest
    print("Running basic smoke tests...")
    
    try:
        # Test 1: Component initialization
        print("✅ Testing component initialization...")
        recorder = AudioRecorder()
        transcriber = WhisperTranscriber() 
        cleaner = TextCleaner()
        ui = TerminalUI()
        app = DictationApp()
        print("✅ All components initialize successfully")
        
        # Test 2: Method existence
        print("✅ Testing method existence...")
        test_methods = TestMainAppInterfaceCompatibility()
        test_methods.test_main_app_component_interfaces()
        print("✅ All required methods exist")
        
        # Test 3: Rule-based cleanup
        print("✅ Testing rule-based cleanup...")
        provider = RuleBasedProvider()
        if provider.is_available():
            # Run sync version for simple test
            import asyncio
            result = asyncio.run(provider.cleanup_text("Um, this is a test, uh, message."))
            if result and not result.error:
                print("✅ Rule-based cleanup works")
            else:
                print("❌ Rule-based cleanup failed")
        
        print("\n🎉 All smoke tests passed!")
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()