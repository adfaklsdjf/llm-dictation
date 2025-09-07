"""
Text cleanup orchestration.

Coordinates multiple cleanup providers to clean up raw transcription text,
providing different approaches and allowing user comparison/selection.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .providers import (
    CleanupProvider, 
    OpenAIProvider, 
    ClaudeProvider,
    LocalProvider,
    RuleBasedProvider,
    CleanupResult
)

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class MultiCleanupResult:
    """Result from multiple cleanup providers."""
    original_text: str
    results: List[CleanupResult]
    best_result: Optional[CleanupResult] = None
    processing_time: float = 0.0
    strategy: Optional[str] = None


class TextCleaner:
    """
    Orchestrates text cleanup using multiple providers.
    
    Manages provider selection, parallel processing, and result comparison
    to provide users with multiple cleaned-up versions of their transcription.
    """
    
    def __init__(self, providers: List[CleanupProvider] = None):
        """
        Initialize the text cleaner.
        
        Args:
            providers: List of cleanup providers to use. If None, uses default providers.
        """
        self.providers = providers or self._get_default_providers()
        self._enabled_providers = {provider.name: True for provider in self.providers}
        self.performance_metrics = {}
    
    def _get_default_providers(self) -> List[CleanupProvider]:
        """Get the default set of cleanup providers."""
        providers = []
        
        # Add OpenAI provider if available
        openai_provider = OpenAIProvider()
        if openai_provider.is_available():
            providers.append(openai_provider)
        
        # Add Claude provider if available  
        claude_provider = ClaudeProvider()
        if claude_provider.is_available():
            providers.append(claude_provider)
            
        # Add local provider (placeholder)
        local_provider = LocalProvider()
        if local_provider.is_available():
            providers.append(local_provider)
        
        # Always add rule-based provider as fallback
        providers.append(RuleBasedProvider())
        
        return providers
    
    async def cleanup_text(
        self,
        raw_text: str,
        strategy: str = "parallel"
    ) -> List[CleanupResult]:
        """
        Clean up text using multiple providers.
        
        Args:
            raw_text: Raw text to clean up
            strategy: Cleanup strategy ('parallel', 'cascade', 'single')
            
        Returns:
            List of CleanupResult from providers.
        """
        if not raw_text.strip():
            return []
        
        start_time = time.time()
        
        # Select available providers
        active_providers = [
            p for p in self.providers 
            if self._enabled_providers.get(p.name, True) and p.is_available()
        ]
        
        if not active_providers:
            logger.warning("No providers available for text cleanup")
            return []
        
        if strategy == "single":
            # Use only the first available provider
            active_providers = active_providers[:1]
        elif strategy == "cascade":
            # Try providers in order until one succeeds
            return await self._cleanup_cascade(raw_text, active_providers)
        
        # Default: parallel strategy
        return await self._cleanup_parallel(raw_text, active_providers)
    
    async def _cleanup_parallel(
        self, 
        raw_text: str, 
        providers: List[CleanupProvider]
    ) -> List[CleanupResult]:
        """Run cleanup with all providers in parallel."""
        tasks = []
        for provider in providers:
            task = asyncio.create_task(provider.cleanup_text(raw_text))
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return valid results
            valid_results = []
            for result in results:
                if isinstance(result, CleanupResult):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Provider error: {result}")
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Error in parallel cleanup: {e}")
            return []
    
    async def _cleanup_cascade(
        self, 
        raw_text: str, 
        providers: List[CleanupProvider]
    ) -> List[CleanupResult]:
        """Try providers in order until one succeeds."""
        for provider in providers:
            try:
                result = await provider.cleanup_text(raw_text)
                if result and not result.error:
                    return [result]
            except Exception as e:
                logger.error(f"Provider {provider.name} failed: {e}")
                continue
        
        # If all providers failed, return empty list
        return []

    async def get_best_cleanup(self, raw_text: str) -> CleanupResult:
        """
        Get the best cleanup result from available providers.
        
        Args:
            raw_text: Raw text to clean up
            
        Returns:
            Best CleanupResult based on quality score.
        """
        results = await self.cleanup_text(raw_text, strategy="parallel")
        
        if not results:
            # Return a fallback result with the original text
            return CleanupResult(
                original_text=raw_text,
                cleaned_text=raw_text,
                provider="none",
                processing_time=0.0,
                quality_score=0.0,
                error="No providers available"
            )
        
        # Filter out error results and find the best quality score
        valid_results = [r for r in results if not r.error]
        
        if not valid_results:
            # Return the first result even if it has an error
            return results[0]
        
        # Return result with highest quality score
        best_result = max(valid_results, key=lambda r: r.quality_score)
        return best_result
    
    def add_provider(self, provider: CleanupProvider) -> None:
        """Add a new provider to the cleaner."""
        if provider not in self.providers:
            self.providers.append(provider)
            self._enabled_providers[provider.name] = True
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [p.get_provider_name() for p in self.providers if p.is_available()]
    
    def set_provider_enabled(self, provider_name: str, enabled: bool) -> None:
        """Enable or disable a specific provider."""
        self._enabled_providers[provider_name] = enabled
    
    def is_provider_enabled(self, provider_name: str) -> bool:
        """Check if a provider is enabled."""
        return self._enabled_providers.get(provider_name, False)
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled provider names."""
        return [
            p.get_provider_name() for p in self.providers 
            if p.is_available() and self._enabled_providers.get(p.name, True)
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all providers."""
        metrics = {}
        for provider in self.providers:
            metrics[provider.get_provider_name()] = {
                'usage_stats': provider.get_usage_stats(),
                'available': provider.is_available(),
                'enabled': self._enabled_providers.get(provider.name, True)
            }
        return metrics
    
    async def benchmark_providers(
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
                results[provider.get_provider_name()] = {
                    'available': False,
                    'processing_time': 0.0,
                    'success': False,
                    'quality_score': 0.0,
                    'error': 'Provider not available'
                }
                continue
                
            try:
                start_time = time.time()
                result = await provider.cleanup_text(test_text)
                end_time = time.time()
                
                results[provider.get_provider_name()] = {
                    'available': True,
                    'processing_time': end_time - start_time,
                    'success': result is not None and not result.error,
                    'quality_score': result.quality_score if result else 0.0,
                    'error': result.error if result else None,
                    'cleaned_length': len(result.cleaned_text) if result else 0,
                    'original_length': len(test_text)
                }
                
            except Exception as e:
                results[provider.get_provider_name()] = {
                    'available': True,
                    'processing_time': 0.0,
                    'success': False,
                    'quality_score': 0.0,
                    'error': str(e),
                    'cleaned_length': 0,
                    'original_length': len(test_text)
                }
        
        return results


def create_default_cleaner() -> TextCleaner:
    """Create a TextCleaner with default configuration."""
    return TextCleaner()


async def quick_cleanup(text: str) -> str:
    """
    Quick text cleanup using the best available provider.
    
    Args:
        text: Text to clean up
        
    Returns:
        Cleaned up text, or original text if cleanup fails.
    """
    cleaner = create_default_cleaner()
    result = await cleaner.get_best_cleanup(text)
    
    if result and not result.error:
        return result.cleaned_text
    
    return text


# Convenience function for backwards compatibility
async def cleanup_text_with_multiple_providers(
    text: str,
    providers: Optional[List[CleanupProvider]] = None
) -> List[CleanupResult]:
    """
    Convenience function to clean text with multiple providers.
    
    Args:
        text: Raw text to clean up
        providers: List of providers to use (None for defaults)
        
    Returns:
        List of cleanup results from all providers.
    """
    cleaner = TextCleaner(providers)
    return await cleaner.cleanup_text(text, strategy="parallel")