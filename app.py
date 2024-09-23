# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# import streamlit as st
# import os
# from dotenv import load_dotenv

# load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# ## Prompt Template

# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system","You are a helpful assistant. Please response to the user queries"),
#         ("user","Question:{question}")
#     ]
# )
# ## streamlit framework

# st.title('Langchain Demo With LLAMA3 API')
# input_text=st.text_input("Search Anything you want?")

# # ollama LLAma2 LLm 
# llm=Ollama(model="llama3.1")
# output_parser=StrOutputParser()
# chain=prompt|llm|output_parser

# if input_text:
#     st.write(chain.invoke({"question":input_text}))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
import speech_recognition as sr  # For speech recognition
from gtts import gTTS  # Google Text-to-Speech
from pydub import AudioSegment  # For audio processing
from pydub.playback import play  # To play audio
import streamlit.components.v1 as components  # For audio player
import base64

# Load environment variables
load_dotenv()

# Set up Langchain API and tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Set the FFMPEG path using environment variables
os.environ["FFMPEG_EXECUTABLE"] = r"D:\ffmpeg-7.0.2-essentials_build\bin\ffmpeg"

# Set the FFMPEG path using pydub directly (alternative)
AudioSegment.converter = r"D:\ffmpeg-7.0.2-essentials_build\bin\ffmpeg"



# Define the Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework title
st.title('Langchain Demo With LLAMA3 API - Voice Integrated')

# ollama Llama 3.1 model setup
llm = Ollama(model="llama3.1")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak.")
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data)  # Using Google's free STT API
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
            return ""
        except sr.RequestError:
            st.error("Request error from Google Speech Recognition")
            return ""

# Function to convert text to speech
def text_to_speech(response_text):
    tts = gTTS(text=response_text, lang='en')
    tts.save("response.mp3")
    audio = AudioSegment.from_mp3("response.mp3")
    return audio

# Function to play audio in Streamlit
def play_audio(audio_file):
    audio_bytes = audio_file.export(format="mp3")
    b64 = base64.b64encode(audio_bytes.read()).decode()
    audio_html = f"""
        <audio autoplay="true" controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Option to choose input method (Text or Voice)
input_choice = st.radio("Choose input method:", ('Text', 'Voice'))

# Handle text input
if input_choice == 'Text':
    input_text = st.text_input("Search Anything you want?")
    if input_text:
        # Send text input to the chatbot
        response = chain.invoke({"question": input_text})
        st.write(response)
        # Convert chatbot's response to speech
        audio_response = text_to_speech(response)
        # Play the audio response
        play_audio(audio_response)

# Handle voice input
elif input_choice == 'Voice':
    if st.button('Record Voice'):
        # Convert speech to text
        input_text = speech_to_text()
        if input_text:
            # Send the transcribed text to the chatbot
            response = chain.invoke({"question": input_text})
            st.write(response)
            # Convert chatbot's response to speech
            audio_response = text_to_speech(response)
            # Play the audio response
            play_audio(audio_response)
