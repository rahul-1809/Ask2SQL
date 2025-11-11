#!/bin/bash
# Quick start script for the Text-to-SQL chatbot
# Usage: ./run.sh

# Check if GROQ_API_KEY is set, otherwise prompt user
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  GROQ_API_KEY environment variable is not set!"
    echo "Please set it before running:"
    echo "  export GROQ_API_KEY='your-api-key-here'"
    echo ""
    echo "Get your API key from: https://console.groq.com/keys"
    exit 1
fi

# Optional: Set Gemini API key as fallback
# export GEMINI_API_KEY="your-gemini-key"

# Set Flask app
export FLASK_APP=app.app

# Activate virtual environment and run Flask
source .venv/bin/activate
python -m flask run --debug
