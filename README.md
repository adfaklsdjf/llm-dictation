# LLM Dictation

A voice-to-text system that uses local speech recognition and LLM-powered text cleanup to create clean, coherent prompts for language models.

## The Problem

When dictating thoughts to LLMs, raw speech-to-text output often contains:
- Stammering and filler words ("um", "uh", "like")
- Stream-of-consciousness meandering
- Incomplete sentences and false starts
- Repetitions and backtracking

This makes the resulting prompts less effective and harder to read.

## The Solution

**LLM Dictation** creates a pipeline that:
1. **Captures** your speech using local audio recording
2. **Transcribes** it using high-quality local speech-to-text (Whisper)
3. **Cleans up** the raw transcription using LLMs to remove stammering and improve clarity
4. **Delivers** clean, coherent text ready to use as LLM prompts

## Current Status: MVP Development

We're starting with a simple terminal-based proof-of-concept to validate the core workflow and user experience.

### MVP Features
- Terminal app for quick iteration
- Local Whisper for speech-to-text
- OpenAI API for text cleanup
- Clipboard integration for easy use
- Multiple cleanup options to compare results

## Getting Started

*Installation and usage instructions will be added as the MVP is developed.*

## Future Vision

See [VISION.md](VISION.md) for the full roadmap including GUI integration, system-wide hotkeys, and advanced features.