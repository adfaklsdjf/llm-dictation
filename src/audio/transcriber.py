"""
Speech-to-text transcription using Faster Whisper.

Provides optimized Whisper model integration for converting audio
files to text with configurable model sizes and settings.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import time

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed text with timing information."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str     # Transcribed text
    confidence: Optional[float] = None  # Confidence score if available


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    text: str                           # Full transcribed text
    segments: List[TranscriptionSegment] # Individual segments
    language: Optional[str] = None       # Detected language
    duration: Optional[float] = None     # Audio duration
    processing_time: Optional[float] = None  # Time taken to transcribe


class WhisperTranscriber:
    """
    Faster Whisper-based speech-to-text transcriber.
    
    Supports multiple model sizes and provides both full text and
    segmented transcription with timing information.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto"
    ):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run on ('cpu', 'cuda', or 'auto')
            compute_type: Computation precision ('int8', 'float16', 'float32', or 'auto')
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None
        self._model_loaded = False
    
    def load_model(self) -> None:
        """
        Load the Whisper model.
        
        Raises:
            RuntimeError: If Faster Whisper is not available.
            Exception: If model loading fails.
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper not available. Install with: pip install faster-whisper"
            )
        
        if self._model_loaded:
            return
        
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self._model_loaded = True
        except Exception as e:
            raise Exception(f"Failed to load Whisper model '{self.model_size}': {e}")
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Language code to use (None for auto-detection)
            initial_prompt: Initial prompt to guide transcription
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            TranscriptionResult with the transcribed text and metadata.
            
        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If model not loaded or transcription fails.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not self._model_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Transcribe using faster-whisper
            segments, info = self._model.transcribe(
                str(audio_path),
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps
            )
            
            # Convert segments to our format
            transcription_segments = []
            full_text_parts = []
            
            for segment in segments:
                text = segment.text.strip()
                if text:  # Skip empty segments
                    transcription_segments.append(TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=text,
                        confidence=getattr(segment, 'avg_logprob', None)
                    ))
                    full_text_parts.append(text)
            
            processing_time = time.time() - start_time
            full_text = " ".join(full_text_parts)
            
            return TranscriptionResult(
                text=full_text,
                segments=transcription_segments,
                language=info.language if hasattr(info, 'language') else None,
                duration=info.duration if hasattr(info, 'duration') else None,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
    
    def transcribe_streaming(
        self,
        audio_path: Path,
        chunk_callback: Optional[callable] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio with streaming/progressive results.
        
        Args:
            audio_path: Path to the audio file
            chunk_callback: Optional callback for receiving segments as they're processed
            
        Returns:
            Complete TranscriptionResult
        """
        # For now, this is the same as regular transcription
        # In the future, this could provide real-time streaming
        result = self.transcribe(audio_path)
        
        if chunk_callback:
            for segment in result.segments:
                chunk_callback(segment)
        
        return result
    
    def is_model_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'loaded': self._model_loaded,
            'available': FASTER_WHISPER_AVAILABLE
        }


def get_available_models() -> List[str]:
    """
    Get list of available Whisper model sizes.
    
    Returns:
        List of model size strings.
    """
    return ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]


def estimate_model_memory(model_size: str) -> str:
    """
    Estimate memory requirements for a given model size.
    
    Args:
        model_size: Whisper model size
        
    Returns:
        Estimated memory requirement as a string.
    """
    memory_estimates = {
        "tiny": "~39 MB",
        "base": "~74 MB", 
        "small": "~244 MB",
        "medium": "~769 MB",
        "large-v1": "~1550 MB",
        "large-v2": "~1550 MB",
        "large-v3": "~1550 MB"
    }
    
    return memory_estimates.get(model_size, "Unknown")


def benchmark_model(
    model_size: str = "base",
    test_duration: float = 10.0
) -> Dict[str, float]:
    """
    Benchmark a Whisper model with a test audio file.
    
    Args:
        model_size: Model size to benchmark
        test_duration: Duration of test audio to generate
        
    Returns:
        Dictionary with performance metrics.
    """
    # TODO: Implement model benchmarking
    # This would create a test audio file and measure transcription speed
    return {
        'model_size': model_size,
        'audio_duration': test_duration,
        'processing_time': 0.0,
        'real_time_factor': 0.0,
        'memory_usage': 0.0
    }