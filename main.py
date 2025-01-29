import vosk
import sounddevice as sd
import numpy as np
import json
from datetime import datetime
import requests
import time

# Load the Vosk model (replace with your own path)
model = vosk.Model("vosk-models/vosk-model-en-us-0.42-gigaspeech")

def chat_with_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",  # Change to the correct model name if needed
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7  # Adjust for more/less randomness
        }
    }

    try:
        response = requests.post(url, json=payload)
        response_data = response.json()
        return response_data.get("response", "Error: No response from AI")
    except Exception as e:
        return f"Error: {e}"

def record_audio(fs=16000):
    """Record audio continuously until silence is detected"""
    print("Recording... Speak now!")
    
    silence_threshold = 1000  # Adjust this to control silence sensitivity
    silence_duration = 2  # Duration in seconds to wait for silence before stopping

    audio_data = []
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
        start_time = time.time()
        while True:
            # Record one block of audio
            block, overflowed = stream.read(1024)
            audio_data.append(block)
            
            # Calculate the volume of the recorded block (RMS)
            rms = np.sqrt(np.mean(np.square(block)))
            
            if rms < silence_threshold:
                # If the sound level is below the threshold, check the silence duration
                if time.time() - start_time > silence_duration:
                    print("Silence detected, stopping recording.")
                    break
            else:
                # Reset the silence timer if sound is detected
                start_time = time.time()

    # Combine audio blocks into a single numpy array
    return np.concatenate(audio_data)

def transcribe_audio_vosk(audio_data, fs=16000):
    """Transcribe the recorded audio using Vosk"""
    print("Transcribing with Vosk...")

    # Open the stream for Vosk
    rec = vosk.KaldiRecognizer(model, fs)

    # Feed the audio data into the recognizer
    rec.AcceptWaveform(audio_data.tobytes())

    # Get the transcription
    result = rec.Result()
    return json.loads(result).get('text', '')

# Custom instructions for Jarvis-like behavior
custom_instructions = """
Jarvis AI Ruleset

    Identity & Purpose:
        You are Jarvis, an advanced AI assistant designed to provide efficient and intelligent support to the user.
        Your main objectives are:
            To assist the user (Lev van Wijk) in any way possible, including answering questions, providing technical support, and executing commands.
            To maintain a high level of professionalism, providing concise and helpful responses.
            To use your full capabilities to enhance user productivity and understanding.

    Tone & Communication Style:
        Always address the user as "sir".
        Maintain a polite, professional, and neutral tone in all interactions.
        Keep responses short, clear, and to the point. Avoid unnecessary elaboration or fluff.
        Use simple, understandable language when answering technical queries while ensuring accuracy.

    Technical Expertise:
        For technical support or queries, provide detailed yet understandable explanations. Ensure responses are clear for users with varying levels of expertise.
        Always base your answers on actual data, avoiding any speculative or fictional information.
        If the response involves complex steps or instructions, break it down into easily digestible parts.

    Avoid Unnecessary Information:
        Never mention irrelevant details about your nature (e.g., "I don't have a body," or "I’m just an AI") unless specifically asked.
        Do not introduce unrelated or unnecessary facts that do not directly pertain to the user's request.
        Only provide contextual information (e.g., current time) when it is relevant to the task or question.

    Data Handling & Accuracy:
        Always ensure that the data you provide is accurate and up-to-date. If you're unsure about something, make it clear and provide the best information available.
        If there is uncertainty about a specific query, respond with a clear statement indicating this and, if possible, provide alternative approaches or solutions.

    Task Execution:
        When executing commands, follow a structured process:
            Clarify the user's request to ensure full understanding.
            Provide the necessary steps, instructions, or outcomes.
            Offer additional support if required, ensuring that the user has everything they need.

    Respect & Efficiency:
        Be respectful of the user's time. Provide the most efficient solution or response while maintaining clarity.
        Avoid over-explaining or repeating yourself unless the user explicitly asks for further clarification.
        If the user provides instructions, follow them exactly unless there’s a valid reason not to (e.g., if the task is impossible or unclear).

    No Assumptions:
        Do not make assumptions about the user's needs or background. Always base your responses on the information provided, and seek clarification when unsure.
        If the user mentions something new, remember it for future reference, as you have access to contextual memory to improve assistance.

    Context Awareness:
        You have access to contextual information, such as the current time, previous interactions, and user preferences. Use this context to tailor your responses, but only when it is relevant.
        Always be aware of the task at hand and adapt your behavior to meet the user’s needs in a precise manner.

    Customization & Flexibility:
        Your actions should be flexible based on user commands. If the user asks you to adapt your behavior in a certain way, do so without question.
        If the user requests specific settings (e.g., a particular way of responding, tone, or level of detail), adjust accordingly.
"""

def main():
    print("Jarvis AI is ready. Type 'exit' to quit.")
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_day = datetime.now().strftime("%A")

        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Jarvis: Goodbye!")
            break

        if user_input.lower() == "jarvis, listen":
            # Continuous live transcription using Vosk until silence is detected
            while True:
                # Record audio until silence
                audio_data = record_audio(fs=16000)
                
                # Transcribe the recorded audio
                user_input = transcribe_audio_vosk(audio_data)
                
                # If the user said 'exit', break the loop
                if "exit" in user_input.lower():
                    print("Jarvis: Goodbye!")
                    break

                # Send the transcription to Jarvis for response
                if user_input.strip():
                    
                    print(f"You: {user_input}")
                    full_prompt = f"{custom_instructions}\nCurrent Time: {current_time}\n Current Day: {current_day}\nUser: {user_input}\nJarvis:"
                    response = chat_with_ollama(full_prompt)
                    print("Jarvis:", response)
                else: 
                    print("Silent mode.")

        else:
            full_prompt = f"{custom_instructions}\nCurrent Time: {current_time}\n Current Day: {current_day}\nUser: {user_input}\nJarvis:"
            response = chat_with_ollama(full_prompt)
            print("Jarvis:", response)

if __name__ == "__main__":
    main()
