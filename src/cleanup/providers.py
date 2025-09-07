"""
LLM provider abstractions for text cleanup.

Provides a unified interface for different LLM services (OpenAI, Claude, local models)
to clean up transcribed text with consistent input/output handling.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import os
import time
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
import logging

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

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CleanupResult:
    """Result from a text cleanup operation."""
    original_text: str
    cleaned_text: str
    provider: str
    processing_time: float
    quality_score: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CleanupProvider(ABC):
    """Abstract base class for cleanup providers."""
    
    def __init__(self, name: str):
        self.name = name
        self.usage_stats = {
            'requests': 0,
            'successful': 0,
            'failed': 0,
            'total_cost': 0.0,
            'total_tokens': 0
        }
    
    @abstractmethod
    async def cleanup_text(self, raw_text: str) -> CleanupResult:
        """Clean up the provided text."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API keys, models, etc.)."""
        pass
    
    def update_usage_stats(self, success: bool, cost: float = 0.0, tokens: int = 0):
        """Update usage statistics."""
        self.usage_stats['requests'] += 1
        if success:
            self.usage_stats['successful'] += 1
        else:
            self.usage_stats['failed'] += 1
        self.usage_stats['total_cost'] += cost
        self.usage_stats['total_tokens'] += tokens
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats.copy()
    
    def calculate_quality_score(self, original: str, cleaned: str) -> float:
        """Calculate quality score for cleanup result."""
        # Basic quality scoring based on improvement metrics
        if not original.strip() or not cleaned.strip():
            return 0.0
        
        # Measure improvements
        original_words = len(original.split())
        cleaned_words = len(cleaned.split())
        
        # Filler word reduction score
        filler_words = ['um', 'uh', 'like', 'you know', 'so', 'well', 'actually']
        original_fillers = sum(1 for word in original.lower().split() 
                             if any(filler in word for filler in filler_words))
        cleaned_fillers = sum(1 for word in cleaned.lower().split() 
                            if any(filler in word for filler in filler_words))
        
        filler_reduction = max(0, (original_fillers - cleaned_fillers) / max(1, original_fillers))
        
        # Length optimization (not too short, not too long)
        length_ratio = cleaned_words / max(1, original_words)
        length_score = 1.0 if 0.7 <= length_ratio <= 1.1 else max(0, 1.0 - abs(length_ratio - 0.9))
        
        # Sentence structure (periods, capitals)
        sentences_original = len(re.findall(r'[.!?]', original))
        sentences_cleaned = len(re.findall(r'[.!?]', cleaned))
        structure_score = min(1.0, sentences_cleaned / max(1, max(1, sentences_original)))
        
        # Weighted average of improvement metrics
        quality_score = (
            0.4 * filler_reduction +
            0.3 * length_score +
            0.3 * structure_score
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def get_cleanup_prompt(self) -> str:
        """Get the system prompt for text cleanup."""
        return """You are a professional editor helping to clean up transcribed speech. The text you receive is from speech-to-text software and may contain:

- Filler words (um, uh, like, you know)
- False starts and repetitions
- Run-on sentences
- Missing punctuation
- Informal speech patterns

Your task is to improve the text while preserving the original meaning and intent. Follow these guidelines:

- Remove filler words and false starts
- Fix grammar and sentence structure
- Improve clarity while maintaining natural tone
- Add proper punctuation and capitalization
- Break up run-on sentences
- Keep the core message and style intact
- Preserve technical terms and proper nouns
- Do not add new information or change the meaning

Return only the cleaned text without any additional commentary or formatting."""


class OpenAIProvider(CleanupProvider):
    """OpenAI GPT provider for text cleanup."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        super().__init__("openai")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self._client: Optional[openai.OpenAI] = None
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "OpenAI"
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return OPENAI_AVAILABLE and bool(self.api_key)
    
    async def cleanup_text(self, raw_text: str) -> CleanupResult:
        """Clean up text using OpenAI GPT."""
        start_time = time.time()
        
        if not self.is_available():
            self.update_usage_stats(success=False)
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="openai",
                processing_time=time.time() - start_time,
                quality_score=0.0,
                error="OpenAI not available (missing API key or package)"
            )
        
        try:
            # Run the synchronous OpenAI call in a thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor, 
                    self._sync_cleanup_text, 
                    raw_text, 
                    start_time
                )
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_usage_stats(success=False)
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="openai",
                processing_time=processing_time,
                quality_score=0.0,
                error=str(e)
            )
    
    def _sync_cleanup_text(self, raw_text: str, start_time: float) -> CleanupResult:
        """Synchronous cleanup method to be run in executor."""
        if not self._client:
            self._client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_cleanup_prompt()},
                    {"role": "user", "content": f"Please clean up this transcribed text:\n\n{raw_text}"}
                ],
                max_tokens=min(4000, len(raw_text.split()) * 3),
                temperature=0.3
            )
            
            processing_time = time.time() - start_time
            cleaned_text = response.choices[0].message.content.strip()
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(raw_text, cleaned_text)
            
            # Track usage
            tokens = response.usage.total_tokens if response.usage else 0
            cost = self._estimate_cost(tokens)
            self.update_usage_stats(success=True, cost=cost, tokens=tokens)
            
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=cleaned_text,
                provider="openai",
                processing_time=processing_time,
                quality_score=quality_score,
                metadata={
                    "model": self.model,
                    "tokens_used": tokens,
                    "estimated_cost": cost
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_usage_stats(success=False)
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="openai",
                processing_time=processing_time,
                quality_score=0.0,
                error=str(e)
            )
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on tokens used."""
        # GPT-3.5-turbo pricing (approximate)
        if "gpt-4" in self.model.lower():
            cost_per_1k_tokens = 0.03  # GPT-4 is more expensive
        else:
            cost_per_1k_tokens = 0.002  # GPT-3.5-turbo
        return (tokens / 1000) * cost_per_1k_tokens


class ClaudeProvider(CleanupProvider):
    """Anthropic Claude provider for text cleanup."""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        super().__init__("claude")
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.timeout = timeout
        self._client: Optional[Anthropic] = None
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "Claude"
    
    def is_available(self) -> bool:
        """Check if Claude is available."""
        return ANTHROPIC_AVAILABLE and bool(self.api_key)
    
    async def cleanup_text(self, raw_text: str) -> CleanupResult:
        """Clean up text using Claude."""
        start_time = time.time()
        
        if not self.is_available():
            self.update_usage_stats(success=False)
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="claude",
                processing_time=time.time() - start_time,
                quality_score=0.0,
                error="Claude not available (missing API key or package)"
            )
        
        try:
            # Run the synchronous Claude call in a thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor, 
                    self._sync_cleanup_text, 
                    raw_text, 
                    start_time
                )
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_usage_stats(success=False)
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="claude",
                processing_time=processing_time,
                quality_score=0.0,
                error=str(e)
            )
    
    def _sync_cleanup_text(self, raw_text: str, start_time: float) -> CleanupResult:
        """Synchronous cleanup method to be run in executor."""
        if not self._client:
            self._client = Anthropic(api_key=self.api_key, timeout=self.timeout)
        
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=min(4000, len(raw_text.split()) * 3),
                system=self.get_cleanup_prompt(),
                messages=[
                    {"role": "user", "content": f"Please clean up this transcribed text:\n\n{raw_text}"}
                ]
            )
            
            processing_time = time.time() - start_time
            cleaned_text = response.content[0].text.strip()
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(raw_text, cleaned_text)
            
            # Track usage
            tokens = response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else 0
            cost = self._estimate_cost(tokens)
            self.update_usage_stats(success=True, cost=cost, tokens=tokens)
            
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=cleaned_text,
                provider="claude",
                processing_time=processing_time,
                quality_score=quality_score,
                metadata={
                    "model": self.model,
                    "tokens_used": tokens,
                    "estimated_cost": cost
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_usage_stats(success=False)
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="claude",
                processing_time=processing_time,
                quality_score=0.0,
                error=str(e)
            )
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on tokens used."""
        # Claude pricing (approximate)
        if "opus" in self.model.lower():
            cost_per_1k_tokens = 0.015  # Claude Opus
        elif "sonnet" in self.model.lower():
            cost_per_1k_tokens = 0.003  # Claude Sonnet
        else:
            cost_per_1k_tokens = 0.00025  # Claude Haiku
        return (tokens / 1000) * cost_per_1k_tokens


class LocalProvider(CleanupProvider):
    """Local model provider placeholder for future implementation."""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("local")
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "Local"
    
    def is_available(self) -> bool:
        """Check if local LLM is available."""
        # TODO: Check for local model files, transformers library, etc.
        # For now, this is a placeholder and not available
        return False
    
    async def cleanup_text(self, raw_text: str) -> CleanupResult:
        """Clean up text using local LLM (placeholder)."""
        start_time = time.time()
        
        self.update_usage_stats(success=False)
        return CleanupResult(
            original_text=raw_text,
            cleaned_text=raw_text,
            provider="local",
            processing_time=time.time() - start_time,
            quality_score=0.0,
            error="Local model provider not yet implemented"
        )


class RuleBasedProvider(CleanupProvider):
    """Simple rule-based cleanup provider as fallback."""
    
    def __init__(self):
        super().__init__("rule_based")
        # Common filler words and patterns
        self.filler_words = [
            'um', 'uh', 'like', 'you know', 'so', 'well', 'actually',
            'basically', 'literally', 'i mean', 'you see', 'right'
        ]
        self.false_starts = [
            r'\b(and|but|so|well)\s+\1\b',  # Repeated conjunctions
            r'\b(i|we|they|he|she)\s+\1\b',  # Repeated pronouns
        ]
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "Rule-Based"
    
    def is_available(self) -> bool:
        """Rule-based provider is always available."""
        return True
    
    async def cleanup_text(self, raw_text: str) -> CleanupResult:
        """Clean up text using rule-based approach."""
        start_time = time.time()
        
        try:
            cleaned_text = self._rule_based_cleanup(raw_text)
            quality_score = self.calculate_quality_score(raw_text, cleaned_text)
            
            self.update_usage_stats(success=True)
            
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=cleaned_text,
                provider="rule_based",
                processing_time=time.time() - start_time,
                quality_score=quality_score,
                metadata={"method": "rule-based"}
            )
            
        except Exception as e:
            self.update_usage_stats(success=False)
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="rule_based",
                processing_time=time.time() - start_time,
                quality_score=0.0,
                error=str(e)
            )
    
    def _rule_based_cleanup(self, text: str) -> str:
        """Perform rule-based text cleanup."""
        if not text.strip():
            return text
        
        cleaned = text.lower().strip()
        
        # Remove filler words
        words = cleaned.split()
        filtered_words = []
        
        for i, word in enumerate(words):
            # Clean punctuation for comparison
            clean_word = word.strip('.,!?;:')
            
            # Skip filler words
            if clean_word in self.filler_words:
                continue
                
            # Skip if it's a repeated word (simple false start detection)
            if i > 0 and clean_word == words[i-1].strip('.,!?;:'):
                continue
                
            filtered_words.append(word)
        
        # Rejoin and fix capitalization
        if not filtered_words:
            return text
            
        cleaned = ' '.join(filtered_words)
        
        # Fix repeated patterns using regex
        for pattern in self.false_starts:
            cleaned = re.sub(pattern, r'\1', cleaned, flags=re.IGNORECASE)
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        # Add period if missing and text doesn't end with punctuation
        if cleaned and not re.search(r'[.!?]\s*$', cleaned):
            cleaned += '.'
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()


class MockProvider(CleanupProvider):
    """Mock provider for testing."""
    
    def __init__(self, should_fail: bool = False, delay: float = 0.1):
        super().__init__("mock")
        self.should_fail = should_fail
        self.delay = delay
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "Mock"
    
    def is_available(self) -> bool:
        """Mock is always available unless configured to fail."""
        return not self.should_fail
    
    async def cleanup_text(self, raw_text: str) -> CleanupResult:
        """Mock cleanup that simulates processing."""
        start_time = time.time()
        
        # Simulate processing time
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_fail:
            self.update_usage_stats(success=False)
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="mock",
                processing_time=time.time() - start_time,
                quality_score=0.0,
                error="Mock provider configured to fail"
            )
        
        # Simple mock cleanup - just clean up the text a bit
        cleaned_text = raw_text.strip()
        if cleaned_text and not cleaned_text.endswith(('.', '!', '?')):
            cleaned_text += '.'
        if cleaned_text:
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:] if len(cleaned_text) > 1 else cleaned_text.upper()
        
        quality_score = self.calculate_quality_score(raw_text, cleaned_text)
        self.update_usage_stats(success=True)
        
        return CleanupResult(
            original_text=raw_text,
            cleaned_text=cleaned_text,
            provider="mock",
            processing_time=time.time() - start_time,
            quality_score=quality_score,
            metadata={"mock_version": "1.0"}
        )