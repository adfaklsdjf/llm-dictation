# Vision: The Perfect Dictation Experience

## Pie-in-the-Sky Ideal

The ultimate goal is a seamless, system-wide dictation interface that feels like natural thought-to-text conversion.

### User Experience Flow

1. **Universal Activation**: A system-wide hotkey (e.g., `⌘⇧D`) activates dictation from anywhere on the system
2. **Live Transcription Window**: A focused GUI window appears showing:
   - Real-time transcription as you speak
   - Visual feedback (waveform, speaking indicator)
   - Clean, readable text formatting
3. **Natural Completion**: A completion hotkey (e.g., `Enter` or `⌘Enter`) signals end of dictation
4. **Intelligent Cleanup**: The raw transcription is sent to LLM for cleanup, removing:
   - Filler words and stammering
   - False starts and corrections
   - Stream-of-consciousness meandering
5. **Review & Confirm**: Cleaned text is displayed for review and confirmation
6. **Direct Integration**: Upon confirmation, the window closes and the clean text is directly inserted into whatever application was previously focused

### Advanced Features

**Multi-Model Comparison**
- Compare cleanup results from different LLMs (local and cloud)
- Learn user preferences over time
- A/B test different cleanup strategies

**Context Awareness**
- Detect the target application (Claude Code, web browser, terminal)
- Adapt cleanup style based on context (technical vs casual)
- Remember per-application preferences

**Voice Training**
- Personalized speech recognition for better accuracy
- Custom vocabulary for technical terms
- Speaking pattern adaptation

**Smart Interruption**
- Pause/resume mid-sentence
- Handle phone calls or interruptions gracefully
- Resume context after breaks

## Technical Architecture Vision

```
System Hotkey → GUI Window → Live STT → User Completion Signal
                     ↓
               Raw Transcription → LLM Cleanup Router → Multiple Models
                     ↓                                       ↓
               Results Comparison ← ← ← ← ← ← ← ← ← ← ← ← ← ←
                     ↓
               User Selection → Direct Text Injection → Target App
```

### Platform Integration
- **macOS**: Native Cocoa app with Accessibility API integration
- **Linux**: X11/Wayland compatibility with desktop environment integration
- **Cross-platform**: Shared core logic with platform-specific UI layers

This vision provides the target to work toward while we validate the core experience through rapid MVP iteration.