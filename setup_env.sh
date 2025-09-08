#!/bin/bash
# Setup script for LLM Dictation environment and API keys

echo "🔧 LLM Dictation Environment Setup"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    echo "❌ Please run this script from the llm-dictation project root directory"
    exit 1
fi

echo "📝 Setting up API keys (optional but recommended for best results)"
echo ""

# Function to prompt for API key
prompt_api_key() {
    local service_name="$1"
    local env_var="$2"
    local current_value="${!env_var}"
    
    if [ -n "$current_value" ]; then
        echo "✅ $service_name API key already set"
    else
        echo "🔑 $service_name API Key:"
        echo "   Get your key from: $3"
        echo "   Leave empty to skip (rule-based cleanup will be used as fallback)"
        read -s -p "   Enter key: " api_key
        echo ""
        
        if [ -n "$api_key" ]; then
            # Add to shell profile
            echo "export $env_var=\"$api_key\"" >> ~/.bashrc
            echo "export $env_var=\"$api_key\"" >> ~/.zshrc 2>/dev/null || true
            export "$env_var"="$api_key"
            echo "   ✅ $service_name API key configured"
        else
            echo "   ⏭️  Skipped $service_name API key"
        fi
    fi
    echo ""
}

# Prompt for OpenAI API key
prompt_api_key "OpenAI" "OPENAI_API_KEY" "https://platform.openai.com/api-keys"

# Prompt for Anthropic API key  
prompt_api_key "Anthropic Claude" "ANTHROPIC_API_KEY" "https://console.anthropic.com/"

echo "🐍 Python Environment Check"
echo ""

# Check Python version
python_version=$(python3 --version 2>/dev/null || echo "Not found")
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    echo "   Install Python 3.9+ from https://python.org"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment found"
else
    echo "🔨 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo "✅ Virtual environment created"
    else
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

echo ""
echo "📦 Installing Dependencies"
echo ""

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing core dependencies..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ Core dependencies installed"
    else
        echo "❌ Failed to install some dependencies"
        echo "   You may need to install system dependencies first:"
        echo ""
        echo "   macOS:"
        echo "     brew install portaudio"
        echo ""
        echo "   Ubuntu/Debian:"
        echo "     sudo apt-get install python3-pyaudio portaudio19-dev"
        echo ""
        echo "   Then run: pip install pyaudio"
    fi
else
    echo "❌ requirements.txt not found"
    exit 1
fi

echo ""
echo "🧪 Running Integration Tests"
echo ""

# Run validation if possible
if python3 validate_integration.py; then
    echo ""
    echo "🎉 Setup Complete!"
    echo ""
    echo "To use LLM Dictation:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run the application: python3 -m src.main"
    echo ""
    echo "💡 Tips:"
    echo "   • Make sure your microphone is connected and permissions are granted"
    echo "   • Run from Terminal (not IDE) for macOS microphone permissions"
    echo "   • Start with smaller Whisper models for faster testing: --model-size tiny"
else
    echo ""
    echo "⚠️  Setup completed with some issues"
    echo "   Check the error messages above and resolve dependencies"
    echo "   The rule-based cleanup should still work without API keys"
fi

echo ""
echo "For help, see README.md or run: python3 -m src.main --help"