import streamlit as st
from pydub import AudioSegment
import openai
import tempfile
import os
import requests
import whisper
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

# Azure OpenAI connection details
azure_openai_key = "22ec84421ec24230a3638d1b51e3a7dc"
azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

# Azure Text-to-Speech connection details
azure_tts_key = "22ec84421ec24230a3638d1b51e3a7dc"
azure_tts_region = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

def correct_transcription(text):
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_key
    }

    data = {
        "messages": [{"role": "user", "content": f"Please correct the following transcription, removing any grammatical mistakes, filler words like 'umm' and 'hmm', and make it sound more natural:\n\n{text}"}],
        "max_tokens": 100,
        "temperature": 0.7
    }

    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Failed to connect or retrieve response: {response.status_code} - {response.text}")

def generate_audio(text, output_file):
    speech_config = SpeechConfig(subscription=azure_tts_key, region=azure_tts_region)
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    audio_config = AudioOutputConfig(filename=output_file)
    
    speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_synthesizer.speak_text_async(text).get()
    
    if result.reason == ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif result.reason == ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

def replace_audio(video_file, audio_file):
    video = AudioSegment.from_file(video_file, format="mp4")
    audio = AudioSegment.from_file(audio_file, format="wav")
    
    output_video = video.set_audio(audio)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        output_video.export(temp_video_file.name, format="mp4")
        temp_output_video = temp_video_file.name
    
    return temp_output_video

def main():
    st.title("Video Audio Replacement")
    
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])
    
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name
        
        st.video(temp_video_path)
        
        if st.button("Replace Audio"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                AudioSegment.from_file(temp_video_path).export(temp_audio.name, format="wav")
                transcription = transcribe_audio(temp_audio.name)
            
            corrected_text = correct_transcription(transcription)
            st.write("Corrected Transcription:")
            st.write(corrected_text)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_new_audio:
                generate_audio(corrected_text, temp_new_audio.name)
                output_video = replace_audio(temp_video_path, temp_new_audio.name)
            
            st.success("Audio replaced successfully!")
            st.video(output_video)
            
            os.remove(temp_video_path)
            os.remove(temp_audio.name)
            os.remove(temp_new_audio.name)
            os.remove(output_video)

if __name__ == "__main__":
    main()