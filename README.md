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

### Implemented Features
- 🎤 **Audio Recording**: Cross-platform recording with PyAudio
- 🗣️ **Speech-to-Text**: Optimized Whisper (large-v3-turbo) transcription
- ✨ **Text Cleanup**: Multiple LLM providers (OpenAI, Claude) + rule-based fallback
- 🖥️ **Terminal UI**: Rich-based interface with progress indicators
- 📋 **Clipboard Integration**: Automatic copying of cleaned text
- 🔄 **Multiple Strategies**: Compare results from different cleanup approaches

## Getting Started

### Prerequisites
- **Python 3.9+** (3.11+ recommended)
- **Microphone access** (system permissions required)
- **4GB+ RAM** (for Whisper models)
- **API Keys** (optional, for best cleanup quality):
  - OpenAI API key for GPT-based cleanup
  - Anthropic API key for Claude-based cleanup

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd llm-dictation

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Install PyAudio** (platform-specific):
   ```bash
   # macOS
   brew install portaudio
   pip install pyaudio

   # Ubuntu/Debian
   sudo apt-get install python3-pyaudio

   # Windows
   pip install pyaudio
   ```

3. **Set up API keys** (optional but recommended):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-claude-api-key"
   ```

### Usage

#### Basic Usage
```bash
# Run the application
python3 -m src.main

# Or with specific options
python3 -m src.main --model-size base --strategy parallel
```

#### Command Line Options
- `--model-size`: Whisper model (`tiny`, `base`, `medium`, `large-v3-turbo`)
- `--strategy`: Cleanup strategy (`parallel`, `cascade`, `single`)

#### Workflow
1. **Start**: Press Enter to begin recording
2. **Record**: Speak naturally (filler words are OK!)
3. **Stop**: Press Enter to end recording
4. **Review**: Compare cleanup results from different providers
5. **Select**: Choose your preferred version (number key)
6. **Use**: Text is copied to clipboard automatically

### Example Session
```
🎤 LLM Dictation - Ready to transcribe your thoughts!

Press Enter to start recording...
🔴 Recording... (Press Enter to stop)

🤖 Transcribing with Whisper...
📝 Raw transcription: "Um, so I was like thinking about, uh, creating this..."

✨ Cleanup Results:
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Provider ┃ Cleaned Text                         ┃ Quality ┃ Time  ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ OpenAI   │ I was thinking about creating this... │ 9.2     │ 1.3s  │
│ Claude   │ I'm considering creating this...      │ 8.8     │ 1.1s  │
│ Rules    │ so I was thinking about creating this │ 6.5     │ 0.1s  │
└─────────┴─────────────────────────────────────────┴─────────┴───────┘

Select result (1-3) or 'q' to quit: 1
✅ Selected text copied to clipboard!
```

### Troubleshooting

#### Microphone Issues (macOS)
If you get "No audio input" errors:
1. Run from Terminal (not IDE) to trigger permission prompts
2. Check System Preferences > Security & Privacy > Microphone
3. Ensure Terminal has microphone access

#### Performance Issues
- Use smaller Whisper models (`--model-size tiny`) for faster processing
- Ensure sufficient RAM (4GB+) for larger models
- Consider GPU acceleration for faster transcription

#### API Errors
- Verify API keys are set correctly
- Check network connection
- Rule-based cleanup works without API keys as fallback

## Future Vision

See [VISION.md](VISION.md) for the full roadmap including GUI integration, system-wide hotkeys, and advanced features.
