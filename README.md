# Voice-Controlled Chart Assistant

A voice-controlled AI assistant that uses Groq's Mixtral model for natural language processing and ElevenLabs for text-to-speech conversion.

## Features
- Voice input using microphone
- AI-powered responses using Groq's Mixtral-8x7b model
- Text-to-speech output using ElevenLabs
- Chat history tracking
- Customizable audio input/output device selection

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

3. Run the app:
```bash
streamlit run app3.py
```

## Environment Variables
- `GROQ_API_KEY`: Your Groq API key from https://console.groq.com/keys
- `ELEVENLABS_API_KEY`: Your ElevenLabs API key from https://elevenlabs.io/

## Usage
1. Select your microphone from the dropdown
2. Select your audio output device
3. Click "Start Listening" and speak your question
4. The assistant will respond with both text and voice
