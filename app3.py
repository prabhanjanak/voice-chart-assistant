import streamlit as st
import os
from dotenv import load_dotenv
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
            return response.content
        else:
            st.error(f"Error generating speech: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error in speech generation: {str(e)}")
        return None

def main():
    st.title("🤖 AI Chat Assistant with Voice")
    st.write("This is a text-based version of the assistant that works in the browser.")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Text input for questions
    user_input = st.text_input("💭 Type your question here:", key="user_input")
    
    # Process input when user submits
    if st.button("🚀 Send") or user_input:
        if user_input:
            # Get AI response
            ai_response = get_groq_response(user_input)
            if ai_response:
                # Display the response
                st.write("🤖 Assistant:", ai_response)
                
                # Generate and play audio response
                with st.spinner("🔊 Generating audio response..."):
                    audio_content = generate_speech(ai_response)
                    if audio_content:
                        st.audio(audio_content, format="audio/mp3")
                
                # Update chat history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "assistant": ai_response
                })
                
                # Clear the input box
                st.session_state.user_input = ""
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.write("👤 You:", chat["user"])
            st.write("🤖 Assistant:", chat["assistant"])
            st.write("---")

if __name__ == "__main__":
    main()
