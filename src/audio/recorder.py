"""
Audio recording functionality using PyAudio.

Handles audio capture from the default microphone with configurable
quality settings and real-time monitoring.
"""

import pyaudio
import wave
import threading
from typing import Optional, Callable
from pathlib import Path
import tempfile


class AudioRecorder:
    """
    Records audio from the default microphone to WAV files.
    
    Supports start/stop recording with configurable audio quality
    and optional real-time audio level monitoring.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        format: int = pyaudio.paInt16
    ):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz (16kHz recommended for Whisper)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_size: Size of audio chunks to read at a time
            format: PyAudio format (paInt16 for 16-bit audio)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._recording_thread: Optional[threading.Thread] = None
        self._is_recording = False
        self._frames = []
        self._output_path: Optional[Path] = None
    
    def start_recording(self, output_path: Optional[Path] = None) -> Path:
        """
        Start recording audio.
        
        Args:
            output_path: Path to save the recording. If None, uses a temporary file.
            
        Returns:
            Path to the output file that will contain the recording.
            
        Raises:
            RuntimeError: If already recording or if audio initialization fails.
        """
        if self._is_recording:
            raise RuntimeError("Recording already in progress")
        
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
        
        self._output_path = output_path
        self._frames = []
        self._is_recording = True
        
        # Initialize PyAudio
        self._audio = pyaudio.PyAudio()
        
        # Open audio stream
        self._stream = self._audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Start recording in a separate thread
        self._recording_thread = threading.Thread(target=self._record_audio)
        self._recording_thread.start()
        
        return self._output_path
    
    def stop_recording(self) -> Path:
        """
        Stop recording and save the audio to the output file.
        
        Returns:
            Path to the saved audio file.
            
        Raises:
            RuntimeError: If not currently recording.
        """
        if not self._is_recording:
            raise RuntimeError("Not currently recording")
        
        self._is_recording = False
        
        # Wait for recording thread to finish
        if self._recording_thread:
            self._recording_thread.join()
        
        # Close the stream and PyAudio
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        
        if self._audio:
            self._audio.terminate()
        
        # Save the recorded audio
        self._save_audio()
        
        return self._output_path
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    def get_audio_level(self) -> float:
        """
        Get the current audio level (0.0 to 1.0).
        
        Returns:
            Current audio level as a normalized float.
            Returns 0.0 if not recording.
        """
        # TODO: Implement real-time audio level monitoring
        # This would calculate RMS or peak level from recent audio data
        return 0.0
    
    def _record_audio(self) -> None:
        """Internal method to record audio in a separate thread."""
        try:
            while self._is_recording:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                self._frames.append(data)
        except Exception as e:
            print(f"Recording error: {e}")
            self._is_recording = False
    
    def _save_audio(self) -> None:
        """Internal method to save recorded frames to a WAV file."""
        if not self._output_path or not self._frames:
            return
        
        with wave.open(str(self._output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self._audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self._frames))


def get_available_devices() -> list[dict]:
    """
    Get a list of available audio input devices.
    
    Returns:
        List of dictionaries with device information (name, index, channels, etc.)
    """
    devices = []
    audio = pyaudio.PyAudio()
    
    try:
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Only input devices
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate']
                })
    finally:
        audio.terminate()
    
    return devices


def test_microphone(duration_seconds: float = 2.0) -> bool:
    """
    Test microphone by recording a short sample.
    
    Args:
        duration_seconds: How long to record for the test.
        
    Returns:
        True if microphone test successful, False otherwise.
    """
    try:
        recorder = AudioRecorder()
        output_path = recorder.start_recording()
        
        import time
        time.sleep(duration_seconds)
        
        recorder.stop_recording()
        
        # Check if file was created and has content
        if output_path.exists() and output_path.stat().st_size > 0:
            output_path.unlink()  # Clean up test file
            return True
        
    except Exception as e:
        print(f"Microphone test failed: {e}")
    
    return False