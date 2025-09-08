"""
Main application entry point for LLM Dictation.

This module provides the command-line interface and orchestrates the
audio recording, transcription, and text cleanup pipeline.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
import tempfile
import signal
import time
from contextlib import asynccontextmanager

import click
from rich.console import Console
import pyperclip

from . import __version__
from .audio.recorder import AudioRecorder
from .audio.transcriber import WhisperTranscriber, TranscriptionResult
from .cleanup.cleaner import TextCleaner
from .cleanup.providers import OpenAIProvider, ClaudeProvider
from .ui.terminal import TerminalUI


class DictationApp:
    """
    Main application class that coordinates all components.
    
    Handles the complete pipeline from audio recording through
    text cleanup and user selection.
    """
    
    def __init__(self):
        """Initialize the dictation application."""
        self.ui = TerminalUI()
        self.recorder = AudioRecorder()
        self.transcriber = WhisperTranscriber()
        self.cleaner = TextCleaner()
        self.console = Console()
        
        # Configuration
        self.temp_dir = Path(tempfile.gettempdir()) / "llm-dictation"
        self.temp_dir.mkdir(exist_ok=True)
        self.cleanup_strategy = "parallel"  # Default strategy
        
        # State
        self._recording = False
        self._recording_path: Optional[Path] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        if self._recording:
            self.console.print("\n[yellow]⚠️  Recording interrupted, stopping...[/yellow]")
            # For async cleanup, we'll let the main exception handler deal with it
            self._recording = False
        sys.exit(0)
    
    async def run_dictation_session(self) -> None:
        """
        Run a complete dictation session.
        
        Handles the full workflow from recording to clipboard copy.
        """
        try:
            # Record audio (welcome message shown in prompt_start_recording)
            audio_path = await self._record_audio()
            if not audio_path:
                return
            
            # Transcribe audio
            transcription = await self._transcribe_audio(audio_path)
            if not transcription:
                return
            
            # Show raw transcription
            self.ui.show_transcription(transcription)
            
            # Clean up text with multiple providers
            cleanup_result = await self._cleanup_text(transcription.text)
            if not cleanup_result:
                return
            
            # Show cleanup results and let user choose
            selected_result = self.ui.show_cleanup_results(cleanup_result)
            if not selected_result:
                return
            
            # Copy to clipboard
            self._copy_to_clipboard(selected_result.cleaned_text)
            
            # Show completion message
            self.ui.show_completion(selected_result.cleaned_text)
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Session cancelled by user.[/yellow]")
        except Exception as e:
            self.console.print(f"[red]❌ Error during dictation session: {e}[/red]")
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files()
    
    async def _record_audio(self) -> Optional[Path]:
        """
        Record audio from microphone.
        
        Returns:
            Path to recorded audio file, or None if recording failed.
        """
        try:
            # Start recording with UI feedback
            self.ui.show_recording_start()
            await self.recorder.start_recording()
            self._recording = True
            
            # Wait for user to press Enter to stop
            # Use asyncio-friendly input handling
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, input)  # Run blocking input in thread
            
            # Stop recording and get audio bytes
            audio_data = await self.recorder.stop_recording()
            self._recording = False
            
            self.ui.show_recording_complete()
            
            if audio_data and len(audio_data) > 100:  # Check for meaningful data
                # Save audio data to temporary file
                timestamp = int(time.time())
                audio_path = self.temp_dir / f"recording_{timestamp}.wav"
                self._recording_path = audio_path
                
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                
                return audio_path
            else:
                self.console.print("[red]❌ No audio was recorded.[/red]")
                return None
                
        except Exception as e:
            self.console.print(f"[red]❌ Recording failed: {e}[/red]")
            self._recording = False
            return None
    
    async def _transcribe_audio(self, audio_path: Path) -> Optional[TranscriptionResult]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription result, or None if transcription failed.
        """
        try:
            with self.ui.show_transcription_progress():
                # Read audio data from file
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                
                # Use the new async transcribe method
                result = await self.transcriber.transcribe(audio_data)
                
            if result and result.text.strip():
                return result
            else:
                self.console.print("[red]❌ No speech detected in recording.[/red]")
                return None
                
        except Exception as e:
            self.console.print(f"[red]❌ Transcription failed: {e}[/red]")
            return None
    
    async def _cleanup_text(self, text: str) -> Optional:
        """
        Clean up transcribed text using multiple providers.
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Multi-cleanup result, or None if cleanup failed.
        """
        try:
            self.console.print("✨ [cyan]Cleaning up text with multiple providers...[/cyan]")
            results = await self.cleaner.cleanup_text(
                text,
                strategy=self.cleanup_strategy
            )
            
            if results:
                return results
            else:
                self.console.print("[red]❌ Text cleanup failed.[/red]")
                return None
                
        except Exception as e:
            self.console.print(f"[red]❌ Cleanup failed: {e}[/red]")
            return None
    
    def _copy_to_clipboard(self, text: str) -> None:
        """
        Copy text to system clipboard.
        
        Args:
            text: Text to copy
        """
        try:
            pyperclip.copy(text)
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Could not copy to clipboard: {e}[/yellow]")
            self.console.print("[dim]Selected text:[/dim]")
            self.console.print(f"[green]{text}[/green]")
    
    def _cleanup_temp_files(self) -> None:
        """Clean up temporary audio files."""
        try:
            if self._recording_path and self._recording_path.exists():
                self._recording_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors


@click.command()
@click.version_option(version=__version__)
@click.option(
    '--model-size',
    default='base',
    help='Whisper model size (tiny, base, small, medium, large)',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large'])
)
@click.option(
    '--strategy',
    default='parallel',
    help='Text cleanup strategy',
    type=click.Choice(['parallel', 'cascade', 'single'])
)
def main(model_size: str, strategy: str) -> None:
    """
    LLM Dictation - AI-powered speech-to-text with intelligent cleanup.
    
    Records audio, transcribes it using Whisper, and cleans up the text
    using various LLM providers. Press Enter to start/stop recording.
    """
    try:
        # Create and configure app
        app = DictationApp()
        
        # Configure transcriber model
        app.transcriber.model_size = model_size
        
        # Configure cleanup strategy (validate)
        if strategy not in ["parallel", "cascade", "single"]:
            click.echo(f"Invalid strategy '{strategy}', using 'parallel'")
            strategy = "parallel"
        app.cleanup_strategy = strategy
        
        # Run the application
        asyncio.run(app.run_dictation_session())
        
    except KeyboardInterrupt:
        click.echo("\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()