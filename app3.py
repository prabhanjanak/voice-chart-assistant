import streamlit as st
import os
from dotenv import load_dotenv
import requests
from groq import Groq
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
import json
import queue
import threading
import av

# Load environment variables
load_dotenv()

# Configure ElevenLabs API Key
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    raise ValueError("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in the .env file.")

# Configure Groq API
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key not found. Please set the GROQ_API_KEY in the .env file.")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Initialize session state for audio
if 'audio_buffer' not in st.session_state:
    st.session_state['audio_buffer'] = queue.Queue()

def get_groq_response(question):
    try:
        # Generate response using Groq's Mixtral model
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that provides clear and concise responses."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        # Extract response text
        if completion and completion.choices:
            return completion.choices[0].message.content
        else:
            st.error("No valid response received from Groq API")
            return None
            
    except Exception as e:
        st.error(f"Error from Groq API: {str(e)}")
        return None

def generate_speech(text):
    try:
        # Configure ElevenLabs API request
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # Send request to ElevenLabs API
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Error generating speech: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error in speech generation: {str(e)}")
        return None

def process_audio(frame):
    sound = frame.to_ndarray(format="s16")
    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(sound.tobytes(), sample_rate=48000, sample_width=2)
    
    try:
        text = recognizer.recognize_google(audio_data)
        if text:
            st.session_state['audio_buffer'].put(text)
    except sr.UnknownValueError:
        pass
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
    
    return av.AudioFrame.from_ndarray(sound, format='s16')

def main():
    st.title("🎙️ Voice-Controlled Chart Assistant")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create two columns for the layout
    col1, col2 = st.columns([1, 3])

    with col1:
        st.write("🎤 Voice Input:")
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"video": False, "audio": True},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            audio_processing_frame_callback=process_audio
        )

    # Process voice input
    if webrtc_ctx.state.playing:
        while not st.session_state['audio_buffer'].empty():
            user_input = st.session_state['audio_buffer'].get()
            st.write(f"🗣️ You said: {user_input}")
            
            # Get AI response
            ai_response = get_groq_response(user_input)
            if ai_response:
                st.write("🤖 Assistant:", ai_response)
                
                # Generate and play audio response
                audio_content = generate_speech(ai_response)
                if audio_content:
                    st.audio(audio_content, format="audio/mp3")
                
                # Update chat history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "assistant": ai_response
                })

    # Text input as fallback
    user_input = st.text_input("💭 Or type your question here:")
    if user_input:
        # Get AI response
        ai_response = get_groq_response(user_input)
        if ai_response:
            st.write("🤖 Assistant:", ai_response)
            
            # Generate and play audio response
            audio_content = generate_speech(ai_response)
            if audio_content:
                st.audio(audio_content, format="audio/mp3")
            
            # Update chat history
            st.session_state.chat_history.append({
                "user": user_input,
                "assistant": ai_response
            })

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.write("👤 You:", chat["user"])
            st.write("🤖 Assistant:", chat["assistant"])
            st.write("---")

if __name__ == "__main__":
    main()
