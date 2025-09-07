"""
Audio recording functionality using PyAudio.

Handles audio capture from the default microphone with configurable
quality settings optimized for Whisper transcription.
"""

import asyncio
import io
import logging
import struct
import wave
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

import pyaudio
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class AudioRecorderError(Exception):
    """Base exception for audio recorder errors."""
    pass


class MicrophonePermissionError(AudioRecorderError):
    """Raised when microphone permissions are not granted."""
    pass


class DeviceError(AudioRecorderError):
    """Raised when no audio input devices are available."""
    pass


class AudioRecorder:
    """
    Async audio recorder that captures from microphone and returns WAV bytes.
    
    Optimized for Whisper transcription with 16kHz sample rate, 16-bit depth,
    and mono channel recording. Uses memory-based storage for efficient processing.
    
    Assumptions:
    - Uses PyAudio for cross-platform compatibility
    - Memory-based buffering for MVP simplicity
    - WAV format output with standard headers
    - Graceful error handling with Rich-formatted messages
    - No auto-permission prompting (user handles permissions)
    
    Args:
        sample_rate: Audio sample rate in Hz (16kHz recommended for Whisper)
        chunk_size: Size of audio chunks to read (512 for latency/CPU balance)
        channels: Number of audio channels (1 for mono)
        
    Example:
        >>> recorder = AudioRecorder()
        >>> await recorder.start_recording()
        >>> # ... user speaks ...
        >>> audio_data = await recorder.stop_recording()
        >>> len(audio_data)  # WAV file bytes
        1024
        
        # Using as context manager for automatic cleanup
        >>> async with recorder:
        ...     await recorder.start_recording()
        ...     audio_data = await recorder.stop_recording()
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        channels: int = 1
    ):
        """
        Initialize the audio recorder with Whisper-optimized settings.
        
        Args:
            sample_rate: Sample rate in Hz (16kHz is Whisper's native rate)
            chunk_size: Chunk size in samples (512 balances latency/CPU)
            channels: Number of channels (1 for mono, required by Whisper)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16  # 16-bit audio
        self.sample_width = 2  # 16-bit = 2 bytes per sample
        
        # Internal state
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._is_recording = False
        self._audio_buffer: list[bytes] = []
        self._recording_task: Optional[asyncio.Task] = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate audio configuration parameters."""
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")
        if self.channels not in (1, 2):
            raise ValueError(f"Channels must be 1 or 2, got {self.channels}")
    
    async def start_recording(self) -> None:
        """
        Start recording audio asynchronously.
        
        Initializes PyAudio, opens the audio stream, and begins capturing
        audio data to an internal buffer.
        
        Raises:
            RuntimeError: If already recording
            MicrophonePermissionError: If microphone permissions denied (macOS)
            DeviceError: If no input devices available
            AudioRecorderError: If audio initialization fails
        """
        if self._is_recording:
            raise RuntimeError("Recording already in progress")
        
        try:
            # Initialize PyAudio
            self._audio = pyaudio.PyAudio()
            
            # Check for available input devices
            if not self._has_input_devices():
                raise DeviceError("No audio input devices found")
            
            # Clear previous buffer
            self._audio_buffer.clear()
            
            try:
                # Open audio stream
                self._stream = self._audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=None,  # We'll read manually for better error handling
                    input_device_index=None  # Use default device
                )
                
            except OSError as e:
                if "device" in str(e).lower() or "input" in str(e).lower():
                    # macOS permission or device error
                    error_msg = self._format_permission_error()
                    raise MicrophonePermissionError(error_msg) from e
                raise AudioRecorderError(f"Failed to open audio stream: {e}") from e
            
            # Start recording
            self._is_recording = True
            self._recording_task = asyncio.create_task(self._record_audio_loop())
            
            logger.info(f"Recording started: {self.sample_rate}Hz, {self.channels} channel(s)")
            
        except Exception as e:
            # Cleanup on any error
            await self._cleanup_resources()
            if isinstance(e, (AudioRecorderError, RuntimeError)):
                raise
            raise AudioRecorderError(f"Failed to start recording: {e}") from e
    
    async def stop_recording(self) -> bytes:
        """
        Stop recording and return WAV audio data.
        
        Stops the recording loop, closes audio resources, and returns
        the captured audio as WAV-formatted bytes.
        
        Returns:
            WAV audio data as bytes
            
        Raises:
            RuntimeError: If not currently recording
            AudioRecorderError: If stop operation fails
        """
        if not self._is_recording:
            raise RuntimeError("Not currently recording")
        
        try:
            # Stop recording flag first
            self._is_recording = False
            
            # Wait for recording task to complete
            if self._recording_task:
                await self._recording_task
                self._recording_task = None
            
            # Cleanup audio resources
            await self._cleanup_resources()
            
            # Generate WAV data from buffer
            wav_data = self._create_wav_data()
            
            logger.info(f"Recording stopped: {len(wav_data)} bytes captured")
            return wav_data
            
        except Exception as e:
            # Ensure cleanup on any error
            await self._cleanup_resources()
            if isinstance(e, RuntimeError):
                raise
            raise AudioRecorderError(f"Failed to stop recording: {e}") from e
    
    def is_recording(self) -> bool:
        """
        Check if currently recording.
        
        Returns:
            True if recording is active, False otherwise
        """
        return self._is_recording
    
    async def _record_audio_loop(self) -> None:
        """
        Internal async loop that captures audio data.
        
        Runs in the background while recording is active, reading chunks
        of audio data and storing them in the internal buffer.
        """
        if not self._stream:
            return
            
        try:
            while self._is_recording:
                try:
                    # Read audio data (non-blocking with timeout)
                    data = self._stream.read(
                        self.chunk_size,
                        exception_on_overflow=False  # Prevent crashes on buffer overrun
                    )
                    
                    if data:
                        self._audio_buffer.append(data)
                    
                    # Small async yield to prevent blocking
                    await asyncio.sleep(0.001)  # 1ms yield
                    
                except Exception as e:
                    logger.warning(f"Audio read error: {e}")
                    # Continue recording unless it's a fatal error
                    if "input overflowed" not in str(e).lower():
                        break
                    await asyncio.sleep(0.01)  # Brief pause on overflow
                    
        except Exception as e:
            logger.error(f"Recording loop error: {e}")
        finally:
            logger.debug("Recording loop ended")
    
    def _create_wav_data(self) -> bytes:
        """
        Create WAV file data from captured audio buffer.
        
        Returns:
            Complete WAV file as bytes with proper headers
        """
        if not self._audio_buffer:
            # Return minimal valid WAV file for empty recording
            return self._create_empty_wav()
        
        # Combine all audio chunks
        audio_data = b''.join(self._audio_buffer)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)
        
        return wav_buffer.getvalue()
    
    def _create_empty_wav(self) -> bytes:
        """Create a minimal valid empty WAV file."""
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b'')  # Empty audio data
        
        return wav_buffer.getvalue()
    
    def _has_input_devices(self) -> bool:
        """Check if any audio input devices are available."""
        if not self._audio:
            return False
            
        try:
            for i in range(self._audio.get_device_count()):
                device_info = self._audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    return True
        except Exception as e:
            logger.warning(f"Error checking input devices: {e}")
            
        return False
    
    def _format_permission_error(self) -> str:
        """Format a helpful permission error message for macOS."""
        return (
            "Microphone access denied. On macOS:\n"
            "1. Open System Preferences → Security & Privacy → Privacy\n"
            "2. Select 'Microphone' from the left panel\n"
            "3. Enable microphone access for Terminal or your application\n"
            "4. Restart the application and try again"
        )
    
    async def _cleanup_resources(self) -> None:
        """Clean up PyAudio resources safely."""
        try:
            # Stop and close stream
            if self._stream:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
                self._stream = None
                
            # Terminate PyAudio
            if self._audio:
                self._audio.terminate()
                self._audio = None
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    async def __aenter__(self) -> "AudioRecorder":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        if self._is_recording:
            try:
                await self.stop_recording()
            except Exception as e:
                logger.warning(f"Error stopping recording during cleanup: {e}")
        
        await self._cleanup_resources()


async def get_available_devices() -> list[dict]:
    """
    Get a list of available audio input devices.
    
    Returns:
        List of dictionaries with device information (name, index, channels, etc.)
        
    Example:
        >>> devices = await get_available_devices()
        >>> for device in devices:
        ...     print(f"{device['name']}: {device['channels']} channels")
    """
    devices = []
    audio = None
    
    try:
        audio = pyaudio.PyAudio()
        
        for i in range(audio.get_device_count()):
            try:
                device_info = audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:  # Only input devices
                    devices.append({
                        'index': i,
                        'name': device_info.get('name', 'Unknown'),
                        'channels': device_info.get('maxInputChannels', 0),
                        'sample_rate': int(device_info.get('defaultSampleRate', 0)),
                        'is_default': i == audio.get_default_input_device_info().get('index', -1)
                    })
            except Exception as e:
                logger.warning(f"Error getting device {i} info: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error enumerating audio devices: {e}")
        raise DeviceError(f"Failed to enumerate audio devices: {e}") from e
    finally:
        if audio:
            audio.terminate()
    
    return devices


async def test_microphone(duration_seconds: float = 2.0) -> bool:
    """
    Test microphone by recording a short sample.
    
    Args:
        duration_seconds: How long to record for the test
        
    Returns:
        True if microphone test successful, False otherwise
        
    Example:
        >>> success = await test_microphone(1.0)
        >>> if success:
        ...     print("Microphone working!")
    """
    try:
        async with AudioRecorder() as recorder:
            console.print("[dim]Testing microphone...[/dim]")
            
            # Start recording
            await recorder.start_recording()
            
            # Record for specified duration
            await asyncio.sleep(duration_seconds)
            
            # Stop and get result
            audio_data = await recorder.stop_recording()
            
            # Check if we got meaningful audio data
            # WAV header is ~44 bytes, so anything larger indicates audio content
            success = len(audio_data) > 100
            
            if success:
                console.print("[green]✓ Microphone test successful[/green]")
            else:
                console.print("[red]✗ Microphone test failed - no audio detected[/red]")
            
            return success
            
    except MicrophonePermissionError as e:
        console.print(f"[red]✗ Microphone test failed - Permission denied[/red]")
        console.print(f"[dim]{e}[/dim]")
        return False
    except DeviceError as e:
        console.print(f"[red]✗ Microphone test failed - Device error[/red]")
        console.print(f"[dim]{e}[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Microphone test failed: {e}[/red]")
        logger.error(f"Microphone test error: {e}")
        return False