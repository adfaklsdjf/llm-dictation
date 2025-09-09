"""
Speech-to-text transcription using Faster Whisper.

Provides optimized Whisper model integration for converting audio
files and bytes to text with configurable model sizes and settings.
Supports real-time performance with large-v3-turbo model and VAD filtering.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
import time
import tempfile
import os
import asyncio
import logging
import platform

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

logger = logging.getLogger(__name__)


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
    Faster Whisper-based speech-to-text transcriber optimized for real-time performance.
    
    Features:
    - Large V3 Turbo model by default (5.4x faster than V2)
    - Automatic device detection (CUDA, MPS for Mac, CPU fallback)
    - VAD filtering for better speech detection
    - Memory-efficient processing with proper compute types
    - Support for both file paths and audio bytes
    - Model caching and reuse for performance
    """
    
    # Available models in order of preference for fallback
    AVAILABLE_MODELS = [
        "large-v3-turbo",  # Latest, fastest large model
        "large-v3",       # Previous large model
        "medium",         # Good balance
        "base",           # Fast, decent quality
        "tiny"            # Fastest, lower quality
    ]
    
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, Any]] = None,
        beam_size: int = 5,
        language: Optional[str] = "en"  # Default to English for better performance
    ):
        """
        Initialize the Whisper transcriber with optimized settings.
        
        Args:
            model_size: Whisper model size (default: 'large-v3-turbo' for best performance)
            device: Device to run on ('cpu', 'cuda', 'mps', or 'auto')
            compute_type: Computation precision ('int8', 'float16', 'float32', or 'auto')
            vad_filter: Enable Voice Activity Detection for better speech detection
            vad_parameters: VAD configuration parameters
            beam_size: Beam size for decoding (5 is good balance of speed/accuracy)
            language: Default language for transcription ('en' for English, None for auto-detect)
        """
        self.model_size = model_size
        self.device = self._detect_optimal_device() if device == "auto" else device
        self.compute_type = self._detect_optimal_compute_type() if compute_type == "auto" else compute_type
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {
            "min_silence_duration_ms": 500,  # Better speech detection
            "speech_pad_ms": 400
        }
        self.beam_size = beam_size
        self.language = language
        
        self._model: Optional[WhisperModel] = None
        self._model_loaded = False
        self._model_info: Dict[str, Any] = {}
    
    def _detect_optimal_device(self) -> str:
        """
        Automatically detect the optimal device for the current system.
        
        Note: MPS (Apple Silicon GPU) is not properly supported by faster-whisper,
        so we use CPU for macOS systems even with Apple Silicon.
        
        Returns:
            Optimal device string ('cuda' or 'cpu')
        """
        try:
            # Check for NVIDIA GPU
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
            
        # For macOS (including Apple Silicon), use CPU
        # faster-whisper doesn't properly support MPS despite PyTorch availability
        if platform.system() == "Darwin":
            logger.info("Running on macOS: using CPU device (MPS not supported by faster-whisper)")
            return "cpu"
                
        return "cpu"
    
    def _detect_optimal_compute_type(self) -> str:
        """
        Detect optimal compute type based on device and system memory.
        
        Returns:
            Optimal compute type string
        """
        if self.device == "cuda":
            # Use float16 for GPU to save memory and increase speed
            return "float16"
# MPS removed - not supported by faster-whisper
        else:
            # CPU - use int8 for better performance and lower memory usage
            if PSUTIL_AVAILABLE:
                try:
                    available_memory_gb = psutil.virtual_memory().available / (1024**3)
                    if available_memory_gb < 4:
                        return "int8"
                    else:
                        return "float32"
                except Exception:
                    # Fallback if psutil fails
                    return "int8"
            else:
                # Conservative default without psutil
                return "int8"
    
    def load_model(self) -> None:
        """
        Load the Whisper model with fallback support.
        
        Attempts to load the requested model, falling back to smaller models
        if memory or compatibility issues occur.
        
        Raises:
            RuntimeError: If Faster Whisper is not available or all models fail to load.
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper not available. Install with: pip install faster-whisper"
            )
        
        if self._model_loaded:
            return
        
        # Try requested model first, then fallback to smaller models
        models_to_try = [self.model_size]
        for model in self.AVAILABLE_MODELS:
            if model != self.model_size and model not in models_to_try:
                models_to_try.append(model)
        
        last_error = None
        devices_to_try = [self.device]
        if self.device != "cpu":
            devices_to_try.append("cpu")  # Always try CPU as ultimate fallback
        
        for model_size in models_to_try:
            for device in devices_to_try:
                try:
                    # Adjust compute type based on device
                    compute_type = self.compute_type
                    if device == "cpu" and self.compute_type == "float16":
                        compute_type = "int8"  # CPU doesn't support float16 well
                    
                    logger.info(f"Loading Whisper model: {model_size} on {device} with {compute_type}")
                    
                    # Load model with optimized settings
                    self._model = WhisperModel(
                        model_size,
                        device=device,
                        compute_type=compute_type,
                        download_root=None,  # Use default cache directory
                        local_files_only=False  # Allow downloading if needed
                    )
                    
                    # Update device settings if we had to fall back
                    if device != self.device:
                        logger.info(f"Fell back to device: {device}")
                        self.device = device
                        self.compute_type = compute_type
                
                    # Update actual model size used (in case of fallback)
                    self.model_size = model_size
                    self._model_loaded = True
                    
                    # Store model information
                    self._model_info = {
                        'model_size': model_size,
                        'device': self.device,
                        'compute_type': self.compute_type,
                        'vad_filter': self.vad_filter,
                        'beam_size': self.beam_size,
                        'language': self.language
                    }
                    
                    logger.info(f"Successfully loaded Whisper model: {model_size} on {device}")
                    return
                    
                except Exception as e:
                    last_error = e
                    error_msg = f"Failed to load model '{model_size}' on device '{device}': {e}"
                    logger.warning(error_msg)
                    # Continue to try next device for this model
                    continue
            # If all devices failed for this model, continue to next model
        
        # All models failed to load
        raise RuntimeError(
            f"Failed to load any Whisper model. Last error: {last_error}"
        )
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_data: WAV audio data as bytes
            language: Language code to use (uses instance default if None)
            initial_prompt: Initial prompt to guide transcription
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            TranscriptionResult with the transcribed text and metadata.
            
        Raises:
            RuntimeError: If model not loaded or transcription fails.
            ValueError: If audio_data is empty or invalid.
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
        
        if not self._model_loaded:
            self.load_model()
        
        # Use instance language as default
        effective_language = language or self.language
        
        start_time = time.time()
        temp_file = None
        
        try:
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Configure transcription parameters
            transcribe_params = {
                "audio": temp_file_path,
                "language": effective_language,
                "initial_prompt": initial_prompt,
                "word_timestamps": word_timestamps,
                "beam_size": self.beam_size,
                "vad_filter": self.vad_filter
            }
            
            # Add VAD parameters if VAD is enabled
            if self.vad_filter and self.vad_parameters:
                transcribe_params["vad_parameters"] = self.vad_parameters
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None, 
                lambda: self._model.transcribe(**transcribe_params)
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
            
            # Log performance metrics
            audio_duration = getattr(info, 'duration', 0)
            if audio_duration > 0:
                rtf = processing_time / audio_duration  # Real-time factor
                logger.info(f"Transcription completed: {processing_time:.2f}s for {audio_duration:.2f}s audio (RTF: {rtf:.2f})")
            
            return TranscriptionResult(
                text=full_text,
                segments=transcription_segments,
                language=getattr(info, 'language', effective_language),
                duration=audio_duration,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
    
    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to text.
        
        Args:
            file_path: Path to the audio file to transcribe
            language: Language code to use (uses instance default if None)
            initial_prompt: Initial prompt to guide transcription
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            TranscriptionResult with the transcribed text and metadata.
            
        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If model not loaded or transcription fails.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Read file and transcribe
        with open(file_path, 'rb') as f:
            audio_data = f.read()
            
        return await self.transcribe(
            audio_data=audio_data,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps
        )
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Whisper model sizes.
        
        Returns:
            List of model size strings in order of preference.
        """
        return self.AVAILABLE_MODELS.copy()
    
    def is_model_available(self, model_size: str) -> bool:
        """
        Check if a specific model is available/supported.
        
        Args:
            model_size: Model size to check
            
        Returns:
            True if model is supported, False otherwise
        """
        return model_size in self.AVAILABLE_MODELS
    
    def is_model_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current model.
        
        Returns:
            Dictionary with model information.
        """
        info = {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'vad_filter': self.vad_filter,
            'beam_size': self.beam_size,
            'language': self.language,
            'loaded': self._model_loaded,
            'available': FASTER_WHISPER_AVAILABLE,
            'available_models': self.AVAILABLE_MODELS
        }
        
        # Add loaded model specific information
        if self._model_loaded and self._model_info:
            info.update(self._model_info)
            
        return info
    
    async def warm_up(self) -> None:
        """
        Warm up the model by loading it and running a small test transcription.
        
        This helps reduce latency for the first real transcription.
        """
        if not self._model_loaded:
            self.load_model()
        
        # Create a small silent audio clip for warm-up
        import wave
        import io
        
        # Create 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        silence = b'\x00' * int(sample_rate * duration * 2)  # 16-bit = 2 bytes per sample
        
        # Create WAV data
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silence)
        
        # Run warm-up transcription
        try:
            wav_data = wav_buffer.getvalue()
            await self.transcribe(wav_data)
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")


def get_available_models() -> List[str]:
    """
    Get list of available Whisper model sizes.
    
    Returns:
        List of model size strings in order of preference.
    """
    return WhisperTranscriber.AVAILABLE_MODELS.copy()


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
        "large-v3": "~1550 MB",
        "large-v3-turbo": "~1550 MB"
    }
    
    return memory_estimates.get(model_size, "Unknown")


async def benchmark_model(
    model_size: str = "large-v3-turbo",
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
    try:
        import wave
        import io
        
        # Check for optional dependencies
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil not available")
        
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy not available")
        
        # Create test audio with some noise (more realistic than silence)
        sample_rate = 16000
        samples = int(sample_rate * test_duration)
        
        # Generate white noise at low volume to simulate speech-like audio
        np.random.seed(42)  # Reproducible results
        audio_samples = (np.random.random(samples) * 0.1 * 32767).astype(np.int16)
        
        # Create WAV data
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_samples.tobytes())
        
        wav_data = wav_buffer.getvalue()
        
        # Initialize transcriber
        transcriber = WhisperTranscriber(model_size=model_size)
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Benchmark transcription
        start_time = time.time()
        result = await transcriber.transcribe(wav_data)
        end_time = time.time()
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        processing_time = end_time - start_time
        real_time_factor = processing_time / test_duration if test_duration > 0 else 0
        
        return {
            'model_size': model_size,
            'audio_duration': test_duration,
            'processing_time': processing_time,
            'real_time_factor': real_time_factor,
            'memory_usage_mb': memory_used,
            'transcribed_text_length': len(result.text) if result.text else 0,
            'segments_count': len(result.segments) if result.segments else 0
        }
        
    except ImportError as e:
        logger.warning(f"Benchmark requires additional packages: {e}")
        return {
            'model_size': model_size,
            'audio_duration': test_duration,
            'processing_time': 0.0,
            'real_time_factor': 0.0,
            'memory_usage_mb': 0.0,
            'error': f"Missing dependencies: {e}"
        }
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {
            'model_size': model_size,
            'audio_duration': test_duration,
            'processing_time': 0.0,
            'real_time_factor': 0.0,
            'memory_usage_mb': 0.0,
            'error': str(e)
        }