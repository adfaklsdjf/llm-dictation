"""
Text cleanup orchestration.

Coordinates multiple LLM providers to clean up raw transcription text,
providing different approaches and allowing user comparison/selection.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

from .providers import (
    LLMProvider, 
    OpenAIProvider, 
    ClaudeProvider,
    LocalLLMProvider,
    CleanupResult
)


class CleanupStrategy(Enum):
    """Different text cleanup strategies."""
    CONSERVATIVE = "conservative"  # Minimal changes, preserve original meaning
    BALANCED = "balanced"         # Moderate cleanup, improve clarity
    AGGRESSIVE = "aggressive"     # Extensive cleanup, optimize for readability


@dataclass
class MultiCleanupResult:
    """Result from multiple cleanup providers."""
    original_text: str
    results: List[CleanupResult]
    best_result: Optional[CleanupResult] = None
    processing_time: float = 0.0
    strategy: Optional[CleanupStrategy] = None


class TextCleaner:
    """
    Orchestrates text cleanup using multiple LLM providers.
    
    Manages provider selection, parallel processing, and result comparison
    to provide users with multiple cleaned-up versions of their transcription.
    """
    
    def __init__(self, providers: Optional[List[LLMProvider]] = None):
        """
        Initialize the text cleaner.
        
        Args:
            providers: List of LLM providers to use. If None, uses default providers.
        """
        self.providers = providers or self._get_default_providers()
        self._enabled_providers = {provider.name: True for provider in self.providers}
    
    def _get_default_providers(self) -> List[LLMProvider]:
        """Get the default set of LLM providers."""
        providers = []
        
        # Add OpenAI provider if available
        try:
            providers.append(OpenAIProvider())
        except Exception:
            pass  # API key not available
        
        # Add Claude provider if available  
        try:
            providers.append(ClaudeProvider())
        except Exception:
            pass  # API key not available
            
        # Add local LLM provider as fallback
        try:
            providers.append(LocalLLMProvider())
        except Exception:
            pass  # Local model not available
        
        return providers
    
    def cleanup_text(
        self,
        text: str,
        strategy: CleanupStrategy = CleanupStrategy.BALANCED,
        max_providers: Optional[int] = None,
        timeout: float = 30.0
    ) -> MultiCleanupResult:
        """
        Clean up text using multiple providers.
        
        Args:
            text: Raw text to clean up
            strategy: Cleanup strategy to use
            max_providers: Maximum number of providers to use (None for all)
            timeout: Timeout in seconds for each provider
            
        Returns:
            MultiCleanupResult with results from all providers.
        """
        if not text.strip():
            return MultiCleanupResult(
                original_text=text,
                results=[],
                processing_time=0.0,
                strategy=strategy
            )
        
        start_time = time.time()
        
        # Select providers to use
        active_providers = [
            p for p in self.providers 
            if self._enabled_providers.get(p.name, True) and p.is_available()
        ]
        
        if max_providers:
            active_providers = active_providers[:max_providers]
        
        if not active_providers:
            # No providers available, return original text
            return MultiCleanupResult(
                original_text=text,
                results=[],
                processing_time=time.time() - start_time,
                strategy=strategy
            )
        
        # Process with all providers
        results = []
        for provider in active_providers:
            try:
                result = provider.cleanup_text(text, strategy, timeout)
                if result:
                    results.append(result)
            except Exception as e:
                # Create error result
                error_result = CleanupResult(
                    provider_name=provider.name,
                    original_text=text,
                    cleaned_text=text,  # Fallback to original
                    confidence=0.0,
                    processing_time=0.0,
                    error=str(e)
                )
                results.append(error_result)
        
        processing_time = time.time() - start_time
        
        # Determine best result (highest confidence, no errors)
        best_result = None
        if results:
            valid_results = [r for r in results if not r.error]
            if valid_results:
                best_result = max(valid_results, key=lambda r: r.confidence or 0.0)
        
        return MultiCleanupResult(
            original_text=text,
            results=results,
            best_result=best_result,
            processing_time=processing_time,
            strategy=strategy
        )
    
    async def cleanup_text_async(
        self,
        text: str,
        strategy: CleanupStrategy = CleanupStrategy.BALANCED,
        max_providers: Optional[int] = None,
        timeout: float = 30.0
    ) -> MultiCleanupResult:
        """
        Clean up text asynchronously using multiple providers in parallel.
        
        Args:
            text: Raw text to clean up
            strategy: Cleanup strategy to use
            max_providers: Maximum number of providers to use
            timeout: Timeout in seconds for each provider
            
        Returns:
            MultiCleanupResult with results from all providers.
        """
        if not text.strip():
            return MultiCleanupResult(
                original_text=text,
                results=[],
                processing_time=0.0,
                strategy=strategy
            )
        
        start_time = time.time()
        
        # Select providers
        active_providers = [
            p for p in self.providers 
            if self._enabled_providers.get(p.name, True) and p.is_available()
        ]
        
        if max_providers:
            active_providers = active_providers[:max_providers]
        
        if not active_providers:
            return MultiCleanupResult(
                original_text=text,
                results=[],
                processing_time=time.time() - start_time,
                strategy=strategy
            )
        
        # Create async tasks for each provider
        async def cleanup_with_provider(provider: LLMProvider) -> CleanupResult:
            try:
                # For now, we'll use the sync method
                # TODO: Implement async provider methods
                result = provider.cleanup_text(text, strategy, timeout)
                return result if result else CleanupResult(
                    provider_name=provider.name,
                    original_text=text,
                    cleaned_text=text,
                    confidence=0.0,
                    processing_time=0.0,
                    error="No result returned"
                )
            except Exception as e:
                return CleanupResult(
                    provider_name=provider.name,
                    original_text=text,
                    cleaned_text=text,
                    confidence=0.0,
                    processing_time=0.0,
                    error=str(e)
                )
        
        # Run all providers in parallel
        tasks = [cleanup_with_provider(provider) for provider in active_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to CleanupResult list
        valid_results = []
        for result in results:
            if isinstance(result, CleanupResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                # Handle exceptions that weren't caught in cleanup_with_provider
                valid_results.append(CleanupResult(
                    provider_name="unknown",
                    original_text=text,
                    cleaned_text=text,
                    confidence=0.0,
                    processing_time=0.0,
                    error=str(result)
                ))
        
        processing_time = time.time() - start_time
        
        # Determine best result
        best_result = None
        if valid_results:
            error_free_results = [r for r in valid_results if not r.error]
            if error_free_results:
                best_result = max(error_free_results, key=lambda r: r.confidence or 0.0)
        
        return MultiCleanupResult(
            original_text=text,
            results=valid_results,
            best_result=best_result,
            processing_time=processing_time,
            strategy=strategy
        )
    
    def set_provider_enabled(self, provider_name: str, enabled: bool) -> None:
        """Enable or disable a specific provider."""
        self._enabled_providers[provider_name] = enabled
    
    def is_provider_enabled(self, provider_name: str) -> bool:
        """Check if a provider is enabled."""
        return self._enabled_providers.get(provider_name, False)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [p.name for p in self.providers if p.is_available()]
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled provider names."""
        return [
            p.name for p in self.providers 
            if p.is_available() and self._enabled_providers.get(p.name, True)
        ]
    
    def benchmark_providers(
        self,
        test_text: str = "Um, so like, I was thinking we could, you know, maybe implement this feature."
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark all available providers with test text.
        
        Args:
            test_text: Text to use for benchmarking
            
        Returns:
            Dictionary mapping provider names to performance metrics.
        """
        results = {}
        
        for provider in self.providers:
            if not provider.is_available():
                continue
                
            try:
                start_time = time.time()
                result = provider.cleanup_text(test_text, CleanupStrategy.BALANCED, 30.0)
                end_time = time.time()
                
                results[provider.name] = {
                    'available': True,
                    'processing_time': end_time - start_time,
                    'success': result is not None and not result.error,
                    'confidence': result.confidence if result else 0.0,
                    'error': result.error if result else None
                }
                
            except Exception as e:
                results[provider.name] = {
                    'available': True,
                    'processing_time': 0.0,
                    'success': False,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results


def create_default_cleaner() -> TextCleaner:
    """Create a TextCleaner with default configuration."""
    return TextCleaner()


def quick_cleanup(
    text: str, 
    strategy: CleanupStrategy = CleanupStrategy.BALANCED
) -> str:
    """
    Quick text cleanup using the best available provider.
    
    Args:
        text: Text to clean up
        strategy: Cleanup strategy to use
        
    Returns:
        Cleaned up text, or original text if cleanup fails.
    """
    cleaner = create_default_cleaner()
    result = cleaner.cleanup_text(text, strategy, max_providers=1)
    
    if result.best_result:
        return result.best_result.cleaned_text
    
    return text