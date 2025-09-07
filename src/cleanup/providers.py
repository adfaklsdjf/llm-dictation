"""
LLM provider abstractions for text cleanup.

Provides a unified interface for different LLM services (OpenAI, Claude, local models)
to clean up transcribed text with consistent input/output handling.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import os
import time

# Import statements that may fail if dependencies aren't installed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Re-import from cleaner.py to avoid circular imports
class CleanupStrategy(Enum):
    """Different text cleanup strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class CleanupResult:
    """Result from a text cleanup operation."""
    provider_name: str
    original_text: str
    cleaned_text: str
    confidence: Optional[float] = None  # 0.0 to 1.0, higher is better
    processing_time: float = 0.0
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API keys, models, etc.)."""
        pass
    
    @abstractmethod
    def cleanup_text(
        self,
        text: str,
        strategy: CleanupStrategy = CleanupStrategy.BALANCED,
        timeout: float = 30.0
    ) -> Optional[CleanupResult]:
        """
        Clean up the provided text.
        
        Args:
            text: Raw text to clean up
            strategy: Cleanup strategy to use
            timeout: Maximum time to wait for response
            
        Returns:
            CleanupResult with cleaned text and metadata, or None if failed.
        """
        pass
    
    def get_cleanup_prompt(self, strategy: CleanupStrategy) -> str:
        """Get the system prompt for the specified cleanup strategy."""
        base_prompt = """You are a professional editor helping to clean up transcribed speech. The text you receive is from speech-to-text software and may contain:

- Filler words (um, uh, like, you know)
- False starts and repetitions
- Run-on sentences
- Missing punctuation
- Informal speech patterns

Your task is to improve the text while preserving the original meaning and intent."""
        
        if strategy == CleanupStrategy.CONSERVATIVE:
            return base_prompt + """

CONSERVATIVE approach:
- Make minimal changes
- Only fix obvious transcription errors
- Remove clear filler words (um, uh) but preserve conversational tone
- Add basic punctuation
- Preserve the speaker's voice and style"""
        
        elif strategy == CleanupStrategy.BALANCED:
            return base_prompt + """

BALANCED approach:
- Remove filler words and false starts
- Fix grammar and sentence structure
- Improve clarity while maintaining natural tone
- Add proper punctuation and capitalization
- Break up run-on sentences
- Keep the core message and style intact"""
        
        elif strategy == CleanupStrategy.AGGRESSIVE:
            return base_prompt + """

AGGRESSIVE approach:
- Extensively rewrite for maximum clarity and professionalism
- Remove all conversational elements
- Optimize sentence structure and flow
- Use precise, professional language
- Ensure perfect grammar and punctuation
- Maintain factual accuracy but optimize presentation"""
        
        return base_prompt


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for text cleanup."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None
    ):
        super().__init__("openai")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[openai.OpenAI] = None
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return OPENAI_AVAILABLE and bool(self.api_key)
    
    def cleanup_text(
        self,
        text: str,
        strategy: CleanupStrategy = CleanupStrategy.BALANCED,
        timeout: float = 30.0
    ) -> Optional[CleanupResult]:
        """Clean up text using OpenAI GPT."""
        if not self.is_available():
            return CleanupResult(
                provider_name=self.name,
                original_text=text,
                cleaned_text=text,
                error="OpenAI not available (missing API key or package)"
            )
        
        if not self._client:
            self._client = openai.OpenAI(api_key=self.api_key, timeout=timeout)
        
        start_time = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_cleanup_prompt(strategy)},
                    {"role": "user", "content": f"Please clean up this transcribed text:\n\n{text}"}
                ],
                max_tokens=len(text.split()) * 2,  # Rough estimate
                temperature=0.3  # Lower temperature for more consistent cleanup
            )
            
            processing_time = time.time() - start_time
            cleaned_text = response.choices[0].message.content.strip()
            
            return CleanupResult(
                provider_name=self.name,
                original_text=text,
                cleaned_text=cleaned_text,
                confidence=0.8,  # TODO: Calculate based on response quality
                processing_time=processing_time,
                model_used=self.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                cost_estimate=self._estimate_cost(response.usage.total_tokens if response.usage else 0)
            )
            
        except Exception as e:
            return CleanupResult(
                provider_name=self.name,
                original_text=text,
                cleaned_text=text,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on tokens used."""
        # Rough estimates for GPT-3.5-turbo (as of 2024)
        cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens
        return (tokens / 1000) * cost_per_1k_tokens


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider for text cleanup."""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None
    ):
        super().__init__("claude")
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: Optional[Anthropic] = None
    
    def is_available(self) -> bool:
        """Check if Claude is available."""
        return ANTHROPIC_AVAILABLE and bool(self.api_key)
    
    def cleanup_text(
        self,
        text: str,
        strategy: CleanupStrategy = CleanupStrategy.BALANCED,
        timeout: float = 30.0
    ) -> Optional[CleanupResult]:
        """Clean up text using Claude."""
        if not self.is_available():
            return CleanupResult(
                provider_name=self.name,
                original_text=text,
                cleaned_text=text,
                error="Claude not available (missing API key or package)"
            )
        
        if not self._client:
            self._client = Anthropic(api_key=self.api_key, timeout=timeout)
        
        start_time = time.time()
        
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=len(text.split()) * 2,  # Rough estimate
                system=self.get_cleanup_prompt(strategy),
                messages=[
                    {"role": "user", "content": f"Please clean up this transcribed text:\n\n{text}"}
                ]
            )
            
            processing_time = time.time() - start_time
            cleaned_text = response.content[0].text.strip()
            
            return CleanupResult(
                provider_name=self.name,
                original_text=text,
                cleaned_text=cleaned_text,
                confidence=0.85,  # Claude generally produces high-quality output
                processing_time=processing_time,
                model_used=self.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else None
            )
            
        except Exception as e:
            return CleanupResult(
                provider_name=self.name,
                original_text=text,
                cleaned_text=text,
                processing_time=time.time() - start_time,
                error=str(e)
            )


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using lightweight models."""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("local")
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
    
    def is_available(self) -> bool:
        """Check if local LLM is available."""
        # TODO: Check for local model files, transformers library, etc.
        return False  # Disabled for now
    
    def cleanup_text(
        self,
        text: str,
        strategy: CleanupStrategy = CleanupStrategy.BALANCED,
        timeout: float = 30.0
    ) -> Optional[CleanupResult]:
        """Clean up text using local LLM."""
        if not self.is_available():
            return CleanupResult(
                provider_name=self.name,
                original_text=text,
                cleaned_text=text,
                error="Local LLM not available"
            )
        
        # TODO: Implement local LLM cleanup
        # This would use something like transformers with a lightweight model
        # optimized for text editing tasks
        
        start_time = time.time()
        
        # Placeholder: basic rule-based cleanup
        cleaned_text = self._basic_cleanup(text)
        
        return CleanupResult(
            provider_name=self.name,
            original_text=text,
            cleaned_text=cleaned_text,
            confidence=0.5,  # Lower confidence for basic cleanup
            processing_time=time.time() - start_time,
            model_used="rule-based"
        )
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic rule-based text cleanup as a fallback."""
        # Remove common filler words
        fillers = ["um", "uh", "like", "you know", "so"]
        words = text.split()
        
        cleaned_words = []
        for word in words:
            # Remove filler words (case-insensitive)
            if word.lower().strip(".,!?") not in fillers:
                cleaned_words.append(word)
        
        # Join and capitalize first letter
        cleaned_text = " ".join(cleaned_words)
        if cleaned_text:
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
        
        # Add period if missing
        if cleaned_text and not cleaned_text.endswith((".", "!", "?")):
            cleaned_text += "."
        
        return cleaned_text


class MockProvider(LLMProvider):
    """Mock provider for testing."""
    
    def __init__(self, should_fail: bool = False):
        super().__init__("mock")
        self.should_fail = should_fail
    
    def is_available(self) -> bool:
        """Mock is always available unless configured to fail."""
        return not self.should_fail
    
    def cleanup_text(
        self,
        text: str,
        strategy: CleanupStrategy = CleanupStrategy.BALANCED,
        timeout: float = 30.0
    ) -> Optional[CleanupResult]:
        """Mock cleanup that just adds a prefix."""
        if self.should_fail:
            return CleanupResult(
                provider_name=self.name,
                original_text=text,
                cleaned_text=text,
                error="Mock provider configured to fail"
            )
        
        start_time = time.time()
        
        # Simple mock cleanup
        cleaned_text = f"[CLEANED] {text.strip()}"
        
        return CleanupResult(
            provider_name=self.name,
            original_text=text,
            cleaned_text=cleaned_text,
            confidence=0.9,
            processing_time=time.time() - start_time,
            model_used="mock-v1"
        )