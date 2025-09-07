# Claude Guidelines for LLM Dictation Project

## Pre-Flight Checks

Before embarking on any task, perform a quick pre-flight check:
- Are there any ambiguities in the requirements?
- Are there missing details that could affect the implementation approach?
- Are there any technical concerns or blockers that should be addressed first?

**If any ambiguities, missing details, or concerns are identified, STOP and ask for clarification before proceeding.**

**Note: This applies to the main orchestrating agent only. Sub-agents must follow different guidelines (see below).**

## Sub-Agent vs Main Agent Behavior

### Main Agent (Orchestrator)
- **ALWAYS** perform pre-flight checks and ask clarifying questions
- Wait for user responses before proceeding with ambiguous tasks
- Coordinate between sub-agents and user requirements

### Sub-Agents (Task Executors)
- **CANNOT** ask questions - they only return a single message
- **MUST** make reasonable assumptions and document them
- **MUST** complete their assigned task even with uncertainty
- Refer to SUBAGENT_GUIDELINES.md for default assumptions
- Document assumptions in code comments or task output

## Commit Discipline

Upon completion of any task:
- **ALWAYS commit to git automatically before halting for further instructions**
- There is no harm in committing frequently
- There is potential harm in losing work history by NOT committing
- Use descriptive commit messages that explain what was accomplished

## File Organization

When implementing new functionality:
- **Create new files whenever it's reasonable to do so**
- **Favor creating new files over extending existing ones** if the current file is getting long (300+ lines)
- Keep files focused on a single responsibility when possible
- Use clear, descriptive file names that indicate their purpose

## Project Context

This project is building a voice-to-text system for LLM prompt dictation. The workflow is:
1. Audio capture → Speech-to-text → Text cleanup → LLM integration
2. We're starting with a terminal-based MVP for rapid iteration
3. Future plans include GUI integration and system-wide hotkeys

## Technology Stack (Current)
- Python for rapid prototyping
- Whisper for local speech-to-text
- OpenAI API for text cleanup
- Rich library for terminal UI
- Cross-platform support (macOS priority, Linux future)