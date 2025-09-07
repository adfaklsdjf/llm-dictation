"""
Rich-based terminal user interface.

Provides a beautiful command-line interface for the dictation application
with real-time feedback, progress indicators, and result comparison.
"""

from typing import List, Optional, Callable
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

from ..cleanup.cleaner import MultiCleanupResult, CleanupStrategy
from ..cleanup.providers import CleanupResult
from ..audio.transcriber import TranscriptionResult


class TerminalUI:
    """
    Rich-based terminal interface for the dictation application.
    
    Provides an interactive command-line experience with real-time feedback,
    progress indicators, and side-by-side result comparison.
    """
    
    def __init__(self):
        """Initialize the terminal UI."""
        if not RICH_AVAILABLE:
            raise RuntimeError("Rich library not available. Install with: pip install rich")
        
        self.console = Console()
        self._recording_start_time: Optional[float] = None
    
    def show_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = Text()
        welcome_text.append("🎙️  LLM Dictation", style="bold magenta")
        welcome_text.append("\n\nAI-powered speech-to-text with intelligent cleanup\n")
        
        panel = Panel(
            welcome_text,
            title="Welcome",
            title_align="center",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        self.console.print("\n📋 Instructions:")
        self.console.print("  • Press [bold green]Enter[/bold green] to start recording")
        self.console.print("  • Speak naturally (filler words are okay!)")
        self.console.print("  • Press [bold red]Enter[/bold red] again to stop")
        self.console.print("  • Compare and select your preferred cleanup")
        self.console.print()
    
    def show_recording_start(self) -> None:
        """Show recording started message."""
        self._recording_start_time = time.time()
        
        panel = Panel(
            Text("🔴 RECORDING", style="bold red") + Text("\n\nSpeak now... Press Enter to stop", style="white"),
            title="Recording Audio",
            title_align="center",
            border_style="red",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def show_recording_stop(self) -> None:
        """Show recording stopped message."""
        if self._recording_start_time:
            duration = time.time() - self._recording_start_time
            self.console.print(f"⏹️  Recording stopped ({duration:.1f}s)")
        else:
            self.console.print("⏹️  Recording stopped")
        
        self.console.print()
    
    def show_transcription_progress(self) -> 'TranscriptionProgress':
        """Show transcription progress indicator."""
        return TranscriptionProgress(self.console)
    
    def show_transcription_result(self, result: TranscriptionResult) -> None:
        """Display the raw transcription result."""
        self.console.print("\n📝 Raw Transcription:")
        
        # Create transcription panel
        transcription_text = Text(result.text, style="white")
        
        panel = Panel(
            transcription_text,
            title="Speech-to-Text Result",
            title_align="left",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
        # Show metadata
        if result.language or result.duration or result.processing_time:
            metadata = []
            if result.language:
                metadata.append(f"Language: {result.language}")
            if result.duration:
                metadata.append(f"Duration: {result.duration:.1f}s")
            if result.processing_time:
                metadata.append(f"Processing: {result.processing_time:.1f}s")
            
            self.console.print(f"ℹ️  {' | '.join(metadata)}", style="dim")
        
        self.console.print()
    
    def show_cleanup_progress(self) -> 'CleanupProgress':
        """Show cleanup progress indicator."""
        return CleanupProgress(self.console)
    
    def show_cleanup_results(self, result: MultiCleanupResult) -> Optional[CleanupResult]:
        """
        Display cleanup results and let user select preferred version.
        
        Args:
            result: MultiCleanupResult with all provider results
            
        Returns:
            Selected CleanupResult, or None if user cancelled
        """
        if not result.results:
            self.console.print("❌ No cleanup results available", style="red")
            return None
        
        self.console.print("✨ Cleanup Results:")
        
        # Create table for comparison
        table = Table(
            title="Text Cleanup Comparison",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold white"
        )
        
        table.add_column("#", style="cyan", width=3)
        table.add_column("Provider", style="magenta", width=12)
        table.add_column("Cleaned Text", style="white", width=60)
        table.add_column("Quality", style="green", width=10)
        table.add_column("Time", style="yellow", width=8)
        
        valid_results = []
        for i, cleanup_result in enumerate(result.results):
            if cleanup_result.error:
                # Show error results in red
                table.add_row(
                    str(i + 1),
                    cleanup_result.provider_name,
                    f"[red]Error: {cleanup_result.error}[/red]",
                    "N/A",
                    f"{cleanup_result.processing_time:.1f}s"
                )
            else:
                valid_results.append((i + 1, cleanup_result))
                confidence_str = f"{cleanup_result.confidence:.0%}" if cleanup_result.confidence else "N/A"
                
                table.add_row(
                    str(i + 1),
                    cleanup_result.provider_name,
                    cleanup_result.cleaned_text,
                    confidence_str,
                    f"{cleanup_result.processing_time:.1f}s"
                )
        
        self.console.print(table)
        
        if not valid_results:
            self.console.print("❌ No valid cleanup results", style="red")
            return None
        
        # Highlight best result
        if result.best_result:
            best_index = next(
                (i for i, r in enumerate(result.results) if r == result.best_result),
                None
            )
            if best_index is not None:
                self.console.print(f"\n⭐ Recommended: Option {best_index + 1} ({result.best_result.provider_name})")
        
        self.console.print()
        
        # Get user selection
        while True:
            try:
                choice = Prompt.ask(
                    "Select your preferred version (number) or 'q' to quit",
                    default="1" if result.best_result else None
                )
                
                if choice.lower() == 'q':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(result.results):
                    selected_result = result.results[choice_num - 1]
                    if not selected_result.error:
                        self.console.print(f"✅ Selected: {selected_result.provider_name}", style="green")
                        return selected_result
                    else:
                        self.console.print("❌ Cannot select result with error", style="red")
                else:
                    self.console.print(f"❌ Please enter a number between 1 and {len(result.results)}", style="red")
                    
            except ValueError:
                self.console.print("❌ Please enter a valid number", style="red")
            except KeyboardInterrupt:
                return None
    
    def show_clipboard_success(self, text: str) -> None:
        """Show that text was successfully copied to clipboard."""
        preview = text[:100] + "..." if len(text) > 100 else text
        
        panel = Panel(
            f"📋 Copied to clipboard!\n\n[dim]{preview}[/dim]",
            title="Success",
            title_align="center",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def show_error(self, message: str, title: str = "Error") -> None:
        """Show error message."""
        panel = Panel(
            Text(f"❌ {message}", style="red"),
            title=title,
            title_align="center",
            border_style="red",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def show_info(self, message: str, title: str = "Info") -> None:
        """Show info message."""
        panel = Panel(
            Text(f"ℹ️  {message}", style="blue"),
            title=title,
            title_align="center", 
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def confirm_action(self, message: str) -> bool:
        """Ask user for confirmation."""
        return Confirm.ask(message)
    
    def wait_for_enter(self, message: str = "Press Enter to continue...") -> None:
        """Wait for user to press Enter."""
        Prompt.ask(message, default="")


class TranscriptionProgress:
    """Progress indicator for transcription."""
    
    def __init__(self, console: Console):
        self.console = console
        self._progress = None
        self._task_id = None
    
    def __enter__(self):
        """Start progress display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        self._progress.__enter__()
        self._task_id = self._progress.add_task("🤖 Transcribing audio...", total=None)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress display."""
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, message: str) -> None:
        """Update progress message."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=f"🤖 {message}")


class CleanupProgress:
    """Progress indicator for text cleanup."""
    
    def __init__(self, console: Console):
        self.console = console
        self._progress = None
        self._task_id = None
    
    def __enter__(self):
        """Start progress display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        self._progress.__enter__()
        self._task_id = self._progress.add_task("✨ Cleaning up text...", total=None)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress display."""
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, message: str) -> None:
        """Update progress message."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=f"✨ {message}")


def create_terminal_ui() -> TerminalUI:
    """Create a TerminalUI instance."""
    return TerminalUI()


def demo_ui() -> None:
    """Demo function to show UI capabilities."""
    if not RICH_AVAILABLE:
        print("Rich not available for demo")
        return
    
    ui = TerminalUI()
    
    ui.show_welcome()
    ui.wait_for_enter("Press Enter to see recording demo...")
    
    ui.show_recording_start()
    time.sleep(2)
    ui.show_recording_stop()
    
    # Mock transcription progress
    with ui.show_transcription_progress() as progress:
        progress.update("Loading Whisper model...")
        time.sleep(1)
        progress.update("Processing audio...")
        time.sleep(2)
        progress.update("Generating transcription...")
        time.sleep(1)
    
    ui.show_info("Demo complete!")


if __name__ == "__main__":
    demo_ui()