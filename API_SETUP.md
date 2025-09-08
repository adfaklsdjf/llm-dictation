# API Key Setup Guide

LLM Dictation works with multiple text cleanup providers. API keys are **optional** - the application includes a rule-based fallback that works without any external services.

## Quick Setup

### Option 1: Environment Variables (Recommended)

```bash
# Set for current session
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-claude-key-here"

# Make permanent by adding to your shell profile
echo 'export OPENAI_API_KEY="your-openai-key-here"' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="your-claude-key-here"' >> ~/.bashrc

# For macOS/zsh users:
echo 'export OPENAI_API_KEY="your-openai-key-here"' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY="your-claude-key-here"' >> ~/.zshrc
```

### Option 2: Use Setup Script

```bash
./setup_env.sh
# Follow the interactive prompts
```

## Getting API Keys

### OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Sign in or create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. Use in environment variable: `OPENAI_API_KEY`

**Cost**: ~$0.002 per cleanup (very affordable)

### Anthropic Claude API Key  
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign in or create account
3. Navigate to "API Keys" 
4. Click "Create Key"
5. Copy the key (starts with `sk-ant-`)
6. Use in environment variable: `ANTHROPIC_API_KEY`

**Cost**: ~$0.003 per cleanup (very affordable)

## Testing Your Setup

```bash
# Test that API keys are loaded
python3 -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Claude:', bool(os.getenv('ANTHROPIC_API_KEY')))"

# Run the application
python3 -m src.main
```

## Fallback Options

**Without API keys**, the application will:
- ✅ Still work with rule-based text cleanup
- ✅ Remove common filler words ("um", "uh", "like")
- ✅ Basic grammar and punctuation fixes
- ⚠️ Less sophisticated cleanup than LLM providers

**With API keys**, you get:
- 🌟 Much better text cleanup quality
- 🌟 Context-aware improvements
- 🌟 Multiple provider comparison
- 🌟 Professional-grade results

## Troubleshooting

### "No module named 'openai'" or similar
```bash
pip install -r requirements.txt
```

### API key not recognized
```bash
# Check if key is set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Restart terminal after setting keys
# Or source your profile:
source ~/.bashrc  # or ~/.zshrc
```

### Rate limiting errors
- Both services have generous free tiers
- Typical usage: <$1/month for regular dictation
- Consider using `--strategy single` to use only one provider

### Permission errors (macOS)
- Run from Terminal, not IDE
- Check System Preferences > Security & Privacy > Microphone
- Grant microphone access when prompted

## Usage Examples

```bash
# Use all available providers (default)
python3 -m src.main --strategy parallel

# Use only first available provider (faster)  
python3 -m src.main --strategy single

# Use specific model size for faster processing
python3 -m src.main --model-size tiny --strategy single
```