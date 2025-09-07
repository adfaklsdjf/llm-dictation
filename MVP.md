# MVP: Proof-of-Concept Terminal App

## Goals

Build a fast, simple proof-of-concept that validates the core user experience and technical approach without GUI complexity.

## MVP User Flow

```
1. Run `./dictate` in terminal
2. Press Enter to start recording
3. Speak your prompt naturally (with stammering, filler words, etc.)
4. Press Enter to stop recording
5. See raw transcription output
6. See 2-3 cleaned-up versions from different LLM approaches
7. Select preferred version (number key)
8. Selected text is copied to clipboard
9. Paste wherever needed
```

## Technical Stack

**Core Components:**
- **Python** - Fast prototyping, excellent ML ecosystem
- **OpenAI Whisper** - Proven local speech-to-text
- **pyaudio** - Cross-platform audio recording
- **Rich** - Beautiful terminal UI
- **OpenAI API** - High-quality text cleanup (initially)

**File Structure:**
```
src/
  audio/
    recorder.py      # Audio capture and processing
    transcriber.py   # Whisper integration
  cleanup/
    cleaner.py       # Text cleanup orchestration
    providers.py     # LLM provider abstractions
  ui/
    terminal.py      # Rich-based terminal interface
  main.py            # Application entry point
requirements.txt     # Python dependencies
```

## Key Learning Objectives

**Technical Validation:**
- Whisper accuracy for natural dictation
- Acceptable latency for the full pipeline
- Effective cleanup prompts for different LLM providers
- Audio quality requirements

**UX Validation:**
- Does the workflow feel natural?
- Is real-time transcription necessary, or is batch processing sufficient?
- What level of cleanup is most useful?
- How important is model comparison vs. single best result?

## Implementation Priority

### Phase 1: Basic Pipeline
1. Audio recording with simple start/stop
2. Whisper transcription (local)
3. Single cleanup provider (OpenAI API)
4. Clipboard output

### Phase 2: Comparison & Choice
1. Multiple cleanup providers (local model + cloud)
2. Side-by-side result comparison
3. User selection interface
4. Provider performance metrics

### Phase 3: Polish & Configuration
1. Configuration file for API keys, models
2. Better error handling and user feedback
3. Audio quality indicators
4. Cleanup prompt customization

## Success Criteria

The MVP is successful if:
- **It feels useful** - you actually want to use it for real prompts
- **Quality is acceptable** - cleanup meaningfully improves raw transcription
- **Latency is tolerable** - fast enough that you don't lose your train of thought
- **It reveals next priorities** - shows what to build/improve next

This focused approach lets us validate the core concept quickly and iterate based on real usage patterns.