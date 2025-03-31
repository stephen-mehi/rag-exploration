from pvporcupine import create
import pyaudio
import json
import whisper
import openai
import subprocess
import pyttsx3
from pathlib import Path
import sounddevice as sd
import scipy.io.wavfile as wav
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import soundfile as sf
import numpy as np
import time

import os

porc_api_key = os.getenv("PORC_API_KEY")

os.environ["PATH"] += os.pathsep + "C:\\tools\\ffmpeg\\ffmpeg-2025-03-20-git-76f09ab647-full_build\\ffmpeg-2025-03-20-git-76f09ab647-full_build\\bin"

try:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("FFmpeg is available.")
except FileNotFoundError:
    print("FFmpeg is NOT available to subprocess.")


USE_LOCAL_MODEL = True  # Flip this to switch between GPT-4 and local
REPO_DIRECTORY = "C:\\Users\\SMehi\\source\\repos\\project.cms\\cms-backend" 
porcupine = create(access_key=porc_api_key, keywords=["grasshopper"])
engine = pyttsx3.init()
whisper_model = whisper.load_model("small")

#ASSISTANT CONTEXT*************************
class AssistantContext:
    def __init__(self):
        self.history = []
        self.current_focus = None

    def add_to_history(self, user_text, assistant_reply):
        self.history.append({"user": user_text, "assistant": assistant_reply})
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def set_focus(self, focus):
        self.current_focus = focus

    def get_prompt_context(self):
        context = f"You are assisting with code related to: {self.current_focus}.\n"
        for turn in self.history:
            context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        return context

    def save_to_file(self, path="context.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "history": self.history,
                "current_focus": self.current_focus
            }, f, indent=2)

    def load_from_file(self, path="context.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.history = data.get("history", [])
                self.current_focus = data.get("current_focus")
        except FileNotFoundError:
            pass

def transcribe_audio(audio_path):
    whisper_model = whisper.load_model("medium")
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def speak(text):
    
    engine.say(text)
    engine.runAndWait()

def gpt4_code_gen(prompt):
    openai.api_key = "your-api-key"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're a helpful coding assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def local_code_gen(prompt):
    result = subprocess.run(
        ["ollama", "run", "codellama:13b", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def generate_code(prompt):
    if USE_LOCAL_MODEL:
        return local_code_gen(prompt)
    else:
        return gpt4_code_gen(prompt)


def load_code_files_from_folder(folder_path, extensions=(".cs")):
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extensions):
                full_path = os.path.join(root, file)
                try:
                    print(f"processing file {full_path}...")
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        docs.append(LangchainDocument(
                            page_content=code,
                            metadata={"source": os.path.relpath(full_path, folder_path)}
                        ))
                    print(f"DONE processing file {full_path}...")

                except Exception as e:
                    print(f"Skipped {full_path}: {e}")
    return docs

def RagInProject(folder_path):
        
    # Load documents
    documents = load_code_files_from_folder(folder_path)

    # Smart chunking for code
    # Chunk code with function-aware separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\npublic ",    # C# methods / properties
            "\nprivate ",   # C# internal methods
            "\nprotected ", # C# methods
            "\nclass ",     # C#, Python
            "\nvoid ",      # C# functions
            "\nstatic ",    # C#
            "\ndef ",       # Python
            "\n<",          # XML tag start
            "\n{",          # JSON / C# block start
            "\n\n",         # general paragraph/logical block break
            "\n",           # line-based fallback
            " ",            # word-based fallback
            ""              # character fallback
        ]
    )
    chunks = text_splitter.split_documents(documents)

    # Use BGE Code Embeddings (simple + effective)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    #Connect to local LLM via Ollama
    llm = Ollama(model="codellama")  # or mistral, gemma, etc.

    # RetrievalQA pipeline
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    def ProcessQuery(query): 
        response = qa({"query": query})
        result = response["result"]
        return result
    
    return ProcessQuery



def record_audio_until_wake(fs=16000):
    print("Recording... say 'grasshopper' again to stop.")

    audio_buffer = []
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=512)

    while True:
        pcm = stream.read(512, exception_on_overflow=False)
        audio_buffer.append(pcm)

        # Wake word check
        samples = [int.from_bytes(pcm[i:i+2], 'little', signed=True) for i in range(0, len(pcm), 2)]
        if porcupine.process(samples) >= 0:
            print("Wake word detected again — stopping recording.")
            break


    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Convert to NumPy float32 array
    audio_bytes = b"".join(audio_buffer)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_np = audio_np.reshape(-1, 1)

    return audio_np



def transcribe_audio_from_memory(audio_data, fs=16000):
    if audio_data is None or audio_data.size == 0:
        raise ValueError("Empty or missing audio data")

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    # Save the audio to a regular file on disk
    wav_path = os.path.abspath("last_recording.wav")
    sf.write(wav_path, audio_data.copy(), fs)  # Write a copy to force full flush
    print(f"Transcribing from file: {wav_path}")
    
    time.sleep(0.1)

    if not os.path.exists(wav_path):
        print(wav_path)
        print("File definitely doesn't exist before whisper call.")
    else:
        print("File exists before whisper call.")

    print("Whisper model type:", type(whisper_model))

    result = whisper_model.transcribe(wav_path, verbose=True)

    try:
        os.remove(wav_path)
        print(f"Deleted temporary file: {wav_path}")
    except Exception as e:
        print(f"Warning: failed to delete temp file {wav_path}: {e}")

    return result["text"]

def wait_for_word(audio_stream):
    while True:
        pcm = audio_stream.read(512, exception_on_overflow=False)
        pcm = [int.from_bytes(pcm[i:i+2], 'little', signed=True) for i in range(0, len(pcm), 2)]
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("Word detected!")
            break

def wait_for_word():
    print("Listening for word...")
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
    wait_for_word(stream)
    

def main():
    ctx = AssistantContext()
    ctx.load_from_file()

    output_folder = Path("C:/users/smehi/source/repos/coding_assistant_code_snippets")
    output_folder.mkdir(parents=True, exist_ok=True)

    RagInProject(REPO_DIRECTORY)

    speak("Assistant is ready")
    
    while True:
        # MONITOR FOR WAKE WORD***********************
        #wait_for_word()

        # audio_data = record_audio_until_wake()
        # user_input = transcribe_audio_from_memory(audio_data).strip()
        # print(f"You said: {user_input}")

        # if user_input.lower() == "exit":
        #     speak("Goodbye.")
        #     break

        user_input = "Can you write a method that updates a configuration set's version"
        reply = generate_code(user_input)
        print("Assistant:", reply)

        #speak("Here is the generated code.")

        filename = f"response_{len(ctx.history)}.txt"
        with open(output_folder / filename, "w", encoding="utf-8") as f:
            f.write(reply)

        ctx.save_to_file()

if __name__ == "__main__":
    main()
