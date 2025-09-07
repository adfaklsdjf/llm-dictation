# Sub-Agent Guidelines and Default Assumptions

This file provides default assumptions for sub-agents to prevent deadlocks. Sub-agents cannot receive responses to questions, so they must make reasonable assumptions and document them.

## Core Principles

1. **Complete the task** even with uncertainty
2. **Document assumptions** in code comments or output
3. **Choose simple solutions** for MVP implementations
4. **Follow Python best practices** when in doubt
5. **Prioritize functionality over optimization** initially

## Project Structure Defaults

### Directory Creation
- **Always create full directory structures** as specified in plans
- Use `__init__.py` files for Python packages
- Place source code in `src/` directory as per project conventions

### Dependencies
- **Use latest stable versions** unless specific versions are mentioned
- Include common development dependencies (pytest, black, etc.)
- Create both `requirements.txt` and optional `dev-requirements.txt`

## Audio Implementation Defaults

### Audio Recording Settings
- **Sample Rate**: 16kHz (Whisper's native rate)
- **Bit Depth**: 16-bit
- **Channels**: Mono (1 channel)
- **Chunk Size**: 512 samples (balance of latency/CPU)
- **Format**: WAV output for compatibility

### Audio Library Choices
- **PyAudio** for recording (ecosystem compatibility)
- **Memory-based storage** for MVP (simplicity over optimization)
- **Graceful error handling** with clear messages

### Permissions and Errors
- **macOS**: Provide clear error messages for microphone permissions
- **Cross-platform**: Design for macOS first, Linux compatibility second
- **No auto-prompting**: Simple error messages, let user handle permissions

## Python Code Standards

### Type Hints and Documentation
- **Always use type hints** for function parameters and return values
- **Comprehensive docstrings** with usage examples
- **Document assumptions** in comments when making uncertain choices

### Error Handling
- **Fail gracefully** with informative error messages
- **Use appropriate exceptions** (ValueError, IOError, etc.)
- **Clean up resources** in finally blocks or context managers

### Class Design
- **Single responsibility** per class
- **Configurable parameters** via constructor
- **Proper initialization and cleanup** methods

## File Organization Defaults

### When to Create New Files
- **Always create new files** when specified in plans
- **Split functionality** if a file approaches 300+ lines
- **Use descriptive filenames** that indicate purpose

### Module Organization
```
src/
  audio/
    __init__.py
    recorder.py      # AudioRecorder class
    transcriber.py   # Whisper integration
  cleanup/
    __init__.py
    cleaner.py       # Text cleanup orchestration
    providers.py     # LLM provider abstractions
  ui/
    __init__.py
    terminal.py      # Rich-based terminal interface
  main.py           # Application entry point
```

## MVP-Specific Assumptions

### Terminal UI
- **Rich library** for better terminal experience
- **Simple interactions** - minimize complex user input
- **Clear visual feedback** for ongoing operations

### Text Cleanup
- **OpenAI API** initially (can be replaced later)
- **Simple prompts** for cleanup instructions
- **Fallback options** if API calls fail

### Integration Patterns
- **Loose coupling** between modules
- **Configurable components** via constructor parameters
- **Testable interfaces** with clear boundaries

## Common Implementation Patterns

### Class Templates

```python
from typing import Optional, Any
import logging

class ExampleClass:
    """Brief description of the class.
    
    Assumptions:
    - Document any assumptions made during implementation
    - Note any choices made in absence of specific requirements
    
    Args:
        param1: Description of parameter
        param2: Optional parameter with default
    
    Example:
        >>> example = ExampleClass("value")
        >>> result = example.method()
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        self.param1 = param1
        self.param2 = param2 or 42  # Default assumption documented
        self.logger = logging.getLogger(__name__)
    
    def method(self) -> Any:
        """Method description with clear return type."""
        try:
            # Implementation here
            pass
        except Exception as e:
            self.logger.error(f"Method failed: {e}")
            raise
        finally:
            # Cleanup if needed
            pass
```

### Error Handling Pattern

```python
def robust_operation():
    """Template for operations that might fail."""
    try:
        # Primary operation
        return success_result
    except SpecificError as e:
        # Handle specific known errors
        logger.warning(f"Known issue: {e}")
        return fallback_result
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise RuntimeError(f"Operation failed: {e}") from e
```

## When Uncertain

If you encounter ambiguity not covered here:

1. **Choose the simpler solution**
2. **Document the assumption clearly**
3. **Provide alternative approaches in comments**
4. **Make it easy to change later**

Remember: The goal is to complete tasks and maintain momentum while building something that can be iterated upon.