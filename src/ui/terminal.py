"""
Rich-based terminal user interface.

Provides a beautiful command-line interface for the dictation application
with real-time feedback, progress indicators, and result comparison.
"""

from typing import List, Optional
import asyncio
import time
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

from ..cleanup.providers import CleanupResult


class TerminalUI:
    """
    Rich-based terminal interface for the dictation application.
    
    Provides an async interactive command-line experience with real-time feedback,
    progress indicators, and side-by-side result comparison.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the terminal UI."""
        if not RICH_AVAILABLE:
            raise RuntimeError("Rich library not available. Install with: pip install rich")
        
        self.console = console or Console()
        self._recording_start_time: Optional[float] = None
        self._progress_context = None
        self._current_task = None
    
    async def prompt_start_recording(self) -> bool:
        """
        Prompt user to start recording.
        
        Returns:
            True if user wants to start recording, False otherwise.
        """
        # Show welcome message and instructions
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
        
        # Use asyncio-friendly input
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: input("Press Enter to start recording (or Ctrl+C to quit): ")
            )
            return True
        except KeyboardInterrupt:
            return False
        except EOFError:
            return False
    
    async def show_recording_status(self) -> None:
        """
        Display recording status with visual indicators.
        
        Shows the recording indicator and waits for user to stop.
        """
        self._recording_start_time = time.time()
        
        panel = Panel(
            Text("🔴 RECORDING", style="bold red") + Text("\n\nSpeak now... Press Enter to stop", style="white"),
            title="Recording Audio",
            title_align="center",
            border_style="red",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    async def prompt_stop_recording(self) -> bool:
        """
        Wait for user to stop recording.
        
        Returns:
            True when user presses Enter to stop.
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, input)  # Wait for Enter
            
            # Show recording stopped message
            if self._recording_start_time:
                duration = time.time() - self._recording_start_time
                self.console.print(f"⏹️  Recording stopped ({duration:.1f}s)")
            else:
                self.console.print("⏹️  Recording stopped")
            
            self.console.print()
            return True
        except KeyboardInterrupt:
            return False
        except EOFError:
            return False
    
    async def show_transcription_progress(self, message: str) -> None:
        """
        Display transcription progress with message.
        
        Args:
            message: Progress message to display
        """
        if not self._progress_context:
            self._progress_context = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console
            )
            self._progress_context.__enter__()
            self._current_task = self._progress_context.add_task(f"🤖 {message}", total=None)
        else:
            # Update existing progress
            if self._current_task is not None:
                self._progress_context.update(self._current_task, description=f"🤖 {message}")
        
        # Allow other async operations to run
        await asyncio.sleep(0.1)
    
    def _stop_progress(self):
        """Stop the current progress indicator."""
        if self._progress_context:
            try:
                self._progress_context.__exit__(None, None, None)
            except:
                pass  # Ignore errors during cleanup
            self._progress_context = None
            self._current_task = None
    
    async def display_cleanup_results(self, results: List[CleanupResult]) -> int:
        """
        Display cleanup results and handle user selection.
        
        Args:
            results: List of CleanupResult objects from different providers
            
        Returns:
            Index of selected result (0-based), or -1 if user cancelled
        """
        # Stop any ongoing progress
        self._stop_progress()
        
        if not results:
            await self.show_error(Exception("No cleanup results available"))
            return -1
        
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
        for i, result in enumerate(results):
            if result.error:
                # Show error results in red
                table.add_row(
                    str(i + 1),
                    result.provider,
                    f"[red]Error: {result.error}[/red]",
                    "N/A",
                    f"{result.processing_time:.1f}s"
                )
            else:
                valid_results.append((i, result))
                quality_str = f"{result.quality_score:.1f}" if result.quality_score else "N/A"
                
                table.add_row(
                    str(i + 1),
                    result.provider,
                    result.cleaned_text,
                    quality_str,
                    f"{result.processing_time:.1f}s"
                )
        
        self.console.print(table)
        
        if not valid_results:
            await self.show_error(Exception("No valid cleanup results"))
            return -1
        
        # Highlight best result (highest quality score)
        best_result_idx = -1
        best_quality = 0.0
        for idx, result in valid_results:
            if result.quality_score and result.quality_score > best_quality:
                best_quality = result.quality_score
                best_result_idx = idx
        
        if best_result_idx >= 0:
            self.console.print(f"\n⭐ Recommended: Option {best_result_idx + 1} ({results[best_result_idx].provider})")
        
        self.console.print()
        
        # Get user selection
        while True:
            try:
                loop = asyncio.get_event_loop()
                choice = await loop.run_in_executor(
                    None, 
                    lambda: Prompt.ask(
                        "Select your preferred version (number) or 'q' to quit",
                        default=str(best_result_idx + 1) if best_result_idx >= 0 else "1"
                    )
                )
                
                if choice.lower() == 'q':
                    return -1
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(results):
                    selected_result = results[choice_num - 1]
                    if not selected_result.error:
                        self.console.print(f"✅ Selected: {selected_result.provider_name}", style="green")
                        return choice_num - 1  # Return 0-based index
                    else:
                        self.console.print("❌ Cannot select result with error", style="red")
                else:
                    self.console.print(f"❌ Please enter a number between 1 and {len(results)}", style="red")
                    
            except ValueError:
                self.console.print("❌ Please enter a valid number", style="red")
            except KeyboardInterrupt:
                return -1
            except EOFError:
                return -1
    
    async def show_error(self, error: Exception) -> None:
        """
        Display error message with Rich formatting.
        
        Args:
            error: Exception to display
        """
        # Stop any ongoing progress
        self._stop_progress()
        
        error_message = str(error)
        
        # Provide helpful guidance for common errors
        if "permission" in error_message.lower() or "audio" in error_message.lower():
            guidance = "\n\n💡 Try checking your microphone permissions in System Preferences."
        elif "network" in error_message.lower() or "api" in error_message.lower():
            guidance = "\n\n💡 Check your internet connection and API keys."
        elif "timeout" in error_message.lower():
            guidance = "\n\n💡 Try again - the service might be temporarily slow."
        else:
            guidance = ""
        
        panel = Panel(
            Text(f"❌ {error_message}{guidance}", style="red"),
            title="Error",
            title_align="center",
            border_style="red",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    async def show_success(self, message: str) -> None:
        """
        Display success message with Rich formatting.
        
        Args:
            message: Success message to display
        """
        # Stop any ongoing progress
        self._stop_progress()
        
        # If it looks like clipboard content, show a preview
        if len(message) > 100:
            preview = message[:100] + "..."
            content = f"📋 Copied to clipboard!\n\n[dim]{preview}[/dim]"
        else:
            content = f"✅ {message}"
        
        panel = Panel(
            content,
            title="Success",
            title_align="center",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(panel)


def create_terminal_ui() -> TerminalUI:
    """Create a TerminalUI instance."""
    return TerminalUI()


async def demo_ui() -> None:
    """Demo function to show UI capabilities."""
    if not RICH_AVAILABLE:
        print("Rich not available for demo")
        return
    
    ui = TerminalUI()
    
    # Test the async interface
    try:
        # Test start recording prompt
        start = await ui.prompt_start_recording()
        if not start:
            print("Demo cancelled")
            return
        
        # Test recording status
        await ui.show_recording_status()
        
        # Wait a moment then simulate stop
        await asyncio.sleep(2)
        await ui.prompt_stop_recording()
        
        # Test transcription progress
        await ui.show_transcription_progress("Loading Whisper model...")
        await asyncio.sleep(1)
        await ui.show_transcription_progress("Processing audio...")
        await asyncio.sleep(1)
        await ui.show_transcription_progress("Generating transcription...")
        await asyncio.sleep(1)
        
        # Test cleanup results with mock data
        from ..cleanup.providers import CleanupResult
        
        mock_results = [
            CleanupResult(
                provider="openai",
                original_text="Um, so like, this is a test, you know?",
                cleaned_text="This is a test.",
                quality_score=9.0,
                processing_time=1.2,
                metadata={"model_used": "gpt-3.5-turbo"}
            ),
            CleanupResult(
                provider="claude",
                original_text="Um, so like, this is a test, you know?",
                cleaned_text="This is a test example.",
                quality_score=8.5,
                processing_time=1.5,
                metadata={"model_used": "claude-3-haiku"}
            )
        ]
        
        selected = await ui.display_cleanup_results(mock_results)
        if selected >= 0:
            await ui.show_success(f"Selected result: {mock_results[selected].cleaned_text}")
        
    except KeyboardInterrupt:
        print("\nDemo cancelled")
    except Exception as e:
        await ui.show_error(e)


if __name__ == "__main__":
    asyncio.run(demo_ui())