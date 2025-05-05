import collections
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
import webrtcvad
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = "int16"
CHUNK_SIZE = 2048
VAD_AGGRESSIVENESS = 3
PAUSE_THRESHOLD = 1.0
KEYWORD = "шокин"
PREBUFFER_SECONDS = 1.5

MODELS_PATH = Path("models")

if not MODELS_PATH.exists():
    MODELS_PATH.mkdir()

DATA_PATH = Path("data")
# LOCAL_MODEL_PATH = MODELS_PATH.joinpath(
#     "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
# )
# REMOTE_MODEL = "large-v3-turbo"

# is_downloaded_model = LOCAL_MODEL_PATH.exists()

# MODEL_PATH = LOCAL_MODEL_PATH if is_downloaded_model else REMOTE_MODEL

logger.info(f"CUDA: {torch.cuda.is_available()}")

# Initialize models
model = WhisperModel(
    "large-v3-turbo",  # MODELS_PATH.joinpath("models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"),
    device="cuda",
    compute_type="float16",
    download_root="models",
)
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

with open(DATA_PATH.joinpath("links.json"), "r", encoding="utf-8") as f:
    links_data = json.load(f)

# Initialize the sentence transformer model (multilingual for Russian support)
model_embed = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings for all names in links.json
names = [item["name"] for item in links_data]
# names = ["шокин перейди на " + item["name"] for item in links_data]  # Как вариант костыля
name_embeddings = model_embed.encode(names)


def find_best_link(command):
    """
    Find the link from links.json that best matches the given command based on semantic similarity.

    Args:
        command (str): The user command to match against the names in links.json

    Returns:
        tuple: A tuple containing:
            - str: The most semantically similar link
            - float: The cosine similarity score (between -1 and 1)
    """
    # Generate embedding for the command
    command_embedding = model_embed.encode([command])

    # Calculate cosine similarity between command and all names
    similarities = cosine_similarity(command_embedding, name_embeddings)[0]

    # Find the index of the highest similarity score
    best_index = np.argmax(similarities)

    # Get the best score
    best_score = round(float(similarities[best_index]), 3)

    # Return the corresponding link and score
    return links_data[best_index], best_score


app = FastAPI()


class AudioProcessor:
    def __init__(self):
        self.prebuffer = collections.deque(
            maxlen=int(PREBUFFER_SECONDS * SAMPLE_RATE / (CHUNK_SIZE // 2))
        )
        self.audio_buffer = []
        self.is_recording = False
        self.last_audio_time = time.time()

    async def process_audio(self, data: bytes, websocket: WebSocket):
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(np.square(audio_data / 32768.0)))

        # Update prebuffer
        self.prebuffer.append(data)

        if self.is_recording:
            self.audio_buffer.append(data)
            if rms > 0.005:
                self.last_audio_time = time.time()
            if time.time() - self.last_audio_time > PAUSE_THRESHOLD:
                await self.stop_recording(websocket)
        else:
            if rms > 0.005:
                segments, _ = model.transcribe(
                    audio_data.astype(np.float32) / 32768.0,
                    language="ru",
                    initial_prompt=KEYWORD,
                )
                text = " ".join([seg.text for seg in segments])
                if KEYWORD in text.lower():
                    logger.info(f"Ключевое слово '{KEYWORD}' обнаружено!")
                    self.audio_buffer = list(self.prebuffer)
                    self.audio_buffer.append(data)
                    self.is_recording = True
                    self.last_audio_time = time.time()
                    logger.info("Начало записи...")

    async def stop_recording(self, websocket: WebSocket):
        self.is_recording = False
        full_audio = np.concatenate(
            [np.frombuffer(d, dtype=np.int16) for d in self.audio_buffer]
        )
        segments, _ = model.transcribe(
            full_audio.astype(np.float32) / 32768.0, language="ru", word_timestamps=True
        )
        filtered_text = []
        keyword_found = False
        for segment in segments:
            for word in segment.words:
                if KEYWORD in word.word.lower():
                    keyword_found = True
                    filtered_text = []
                if keyword_found:
                    filtered_text.append(word.word)
        if len(filtered_text) > 2:
            command = " ".join(filtered_text)
            logger.info(f"Распознанный текст: {command}")
            payload, score = find_best_link(command)
            await websocket.send_json(
                {"status": 1, "payload": payload, "transcript": command, "score": score}
            )
        else:
            await websocket.send_json({"status": 0})
        self.audio_buffer = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processor = AudioProcessor()
    try:
        while True:
            data = await websocket.receive_bytes()
            await processor.process_audio(data, websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
