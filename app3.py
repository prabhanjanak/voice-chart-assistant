from dotenv import load_dotenv
import streamlit as st
import os
import speech_recognition as sr
import sounddevice as sd
import plotly.express as px
import requests
from groq import Groq

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
            # Save the audio response to a file
            with open("response.mp3", "wb") as f:
                f.write(response.content)
            return True
        else:
            st.error(f"Error generating speech: {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"Error in speech generation: {str(e)}")
        return False

def main():
    st.title("🎙️ Voice-Controlled Chart Assistant (Groq)")
    
    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    
    # Microphone selection
    st.subheader("🎤 Select Microphone:")
    audio_devices = sd.query_devices()
    input_devices = [device['name'] for device in audio_devices if device['max_input_channels'] > 0]
    selected_input = st.selectbox("", input_devices)
    
    # Audio output selection
    st.subheader("🔊 Select Audio Output:")
    output_devices = [device['name'] for device in audio_devices if device['max_output_channels'] > 0]
    selected_output = st.selectbox("", output_devices)
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Voice input
    if st.button("🎤 Start Listening"):
        with sr.Microphone(device_index=input_devices.index(selected_input)) as source:
            st.write("🎤 Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                st.write("🔄 Processing...")
                
                try:
                    # Convert speech to text
                    user_input = recognizer.recognize_google(audio)
                    st.write(f"🗣️ You said: {user_input}")
                    
                    # Get AI response
                    ai_response = get_groq_response(user_input)
                    if ai_response:
                        st.write(f"🤖 Assistant: {ai_response}")
                        
                        # Generate speech from AI response
                        if generate_speech(ai_response):
                            # Play the audio response
                            with open("response.mp3", "rb") as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format="audio/mp3")
                        
                        # Update chat history
                        st.session_state.chat_history.append({
                            "user": user_input,
                            "assistant": ai_response
                        })
                    
                except sr.UnknownValueError:
                    st.error("Sorry, I couldn't understand what you said.")
                except sr.RequestError as e:
                    st.error(f"Error with the speech recognition service: {e}")
                    
            except sr.WaitTimeoutError:
                st.error("No speech detected within the timeout period.")
    
    # Display chat history
    st.subheader("📝 Chat History:")
    for chat in st.session_state.chat_history:
        st.write(f"👤 User: {chat['user']}")
        st.write(f"🤖 Assistant: {chat['assistant']}")
        st.write("---")

if __name__ == "__main__":
    main()
