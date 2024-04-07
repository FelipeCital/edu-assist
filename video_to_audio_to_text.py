
# video_to_audio_to_text.py

import json
from google.oauth2 import service_account
from google.cloud import speech
from moviepy.editor import VideoFileClip
import io
import streamlit as st

# Retrieve your service account credentials from Streamlit's secrets
gcp_service_account_info = json.loads(st.secrets["gcp_service_account"]["credentials"])
credentials = service_account.Credentials.from_service_account_info(gcp_service_account_info)

# Initialize the Google Cloud client with the credentials
client = speech.SpeechClient(credentials=credentials)

def convert_video_to_audio(video_file_path):
    """Converts a video file to an audio file (WAV format)."""
    clip = VideoFileClip(video_file_path)
    audio_file_path = video_file_path.rsplit('.', 1)[0] + '.wav'
    # Convert to audio (mono channel)
    clip.audio.write_audiofile(audio_file_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
    return audio_file_path

def audio_to_text(audio_file_path):
    """Converts audio file (WAV format) to text using Google Cloud Speech-to-Text API."""
    with io.open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,  # The sample rate must match the file's sample rate
        language_code='en-US',
        audio_channel_count=1,  # Indicate that the audio is mono
        enable_separate_recognition_per_channel=False 
    )

    try:
        response = client.recognize(config=config, audio=audio)
        transcription = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcription
    except Exception as e:
        return f"Could not process the audio file; {e}"

def save_text(text, output_file_path):
    """Saves transcribed text to a file."""
    with open(output_file_path, 'w') as file:
        file.write(text)

