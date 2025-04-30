from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
from faster_whisper import WhisperModel
import librosa
import webrtcvad
import re
from collections import deque

from src.config import ASRConfig
from src.constants import *
from src.vector import vector_search

app = FastAPI()

# Инициализация модели Faster-Whisper
model = WhisperModel(
    model_size_or_path=ASRConfig.get_model_size(),
    device=ASRConfig.get_device(),
    compute_type=ASRConfig.get_compute_type(),
    download_root="models"
)

# Инициализация VAD
vad = webrtcvad.Vad(3)  # Уровень агрессивности: 0 (низкий) - 3 (высокий)

def is_speech(audio_data, sample_rate=OUTPUT_SAMPLE_RATE):
    """Проверка наличия речи с помощью webrtcvad"""
    try:
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        # if sample_rate != OUTPUT_SAMPLE_RATE:
        #     audio_array = librosa.resample(
        #         audio_array,
        #         orig_sr=sample_rate,
        #         target_sr=OUTPUT_SAMPLE_RATE
        #     )
        
        pcm_data = (audio_array * 32768).astype(np.int16)
        frame_samples = int(OUTPUT_SAMPLE_RATE * VAD_FRAME_MS / 1000)
        num_frames = len(pcm_data) // frame_samples
        
        for i in range(num_frames):
            frame = pcm_data[i * frame_samples:(i + 1) * frame_samples].tobytes()
            if vad.is_speech(frame, sample_rate=OUTPUT_SAMPLE_RATE):
                return True
        return False
    except Exception as e:
        print(f"Ошибка в VAD: {e}")
        return False

def check_keyword(fragments):
    """Проверка наличия ключевого слова 'шокин'"""
    try:
        audio_data = b''.join(fragments)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        resampled_array = librosa.resample(
            audio_array,
            orig_sr=INPUT_SAMPLE_RATE,
            target_sr=OUTPUT_SAMPLE_RATE
        )
        segments, _ = model.transcribe(
            resampled_array,
            language="ru",
            beam_size=5,
            initial_prompt=KEYWORD
        )
        transcription = " ".join(segment.text for segment in segments).lower()
        return bool(re.search(rf'\b{KEYWORD}\b', transcription))
    except Exception as e:
        print(f"Ошибка при проверке ключевого слова: {e}")
        return False

def transcribe_audio(fragments):
    """Полная транскрибация аудиофрагментов"""
    try:
        audio_data = b''.join(fragments)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        resampled_array = librosa.resample(
            audio_array,
            orig_sr=INPUT_SAMPLE_RATE,
            target_sr=OUTPUT_SAMPLE_RATE
        )
        segments, _ = model.transcribe(
            resampled_array,
            language="ru",
            beam_size=5,
            initial_prompt=KEYWORD
        )
        transcription = " ".join(segment.text for segment in segments)
        print(f"Транскрибация: {transcription}")
        return transcription
    except Exception as e:
        print(f"Ошибка при транскрибации: {e}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    check_fragments = deque(maxlen=int(CHECK_INTERVAL_MS / (1024 / INPUT_SAMPLE_RATE * 1000)))  # Буфер для проверки
    transcribe_fragments = []  # Буфер для транскрибации
    total_duration_ms = 0      # Длительность всех фрагментов
    check_duration_ms = 0      # Длительность для проверки ключевого слова
    triggered = False         # Флаг активации по ключевому слову
    silence_duration_ms = 0   # Длительность текущей паузы
    
    try:
        while True:
            data = await websocket.receive_bytes()
            check_fragments.append(data)
            
            # Рассчитываем длительность фрагмента
            samples = len(data) // (SAMPLE_WIDTH * CHANNELS)
            fragment_duration_ms = (samples / INPUT_SAMPLE_RATE) * 1000
            total_duration_ms += fragment_duration_ms
            check_duration_ms += fragment_duration_ms
            
            # Проверка на речь с помощью VAD
            is_speech_fragment = is_speech(data, sample_rate=INPUT_SAMPLE_RATE)
            
            # Обработка паузы
            if not is_speech_fragment:
                silence_duration_ms += fragment_duration_ms
            else:
                silence_duration_ms = 0
            
            # Проверка ключевого слова
            if not triggered and check_duration_ms >= CHECK_INTERVAL_MS:
                if check_keyword(check_fragments):
                    print(f"Обнаружено ключевое слово '{KEYWORD}'. Активирован режим транскрибации.")
                    triggered = True
                    # Очищаем transcribe_fragments и копируем check_fragments
                    transcribe_fragments.clear()
                    transcribe_fragments.extend(check_fragments)
                    # await websocket.send_json({
                    #     "info": f"Обнаружено ключевое слово '{KEYWORD}'"
                    # })
                check_duration_ms = 0
            
            # Добавляем новые фрагменты в transcribe_fragments, если режим активирован
            if triggered:
                transcribe_fragments.append(data)
            
            # Полная транскрибация после паузы в 1.5 секунды
            if triggered and silence_duration_ms >= PAUSE_DURATION_MS:
                transcription = transcribe_audio(transcribe_fragments)
                nearest_point = vector_search.find_nearest(query=transcription)
                score = nearest_point.get("score")
                payload = nearest_point.get("payload")
                name = payload.get("name")         
                link = payload.get("link")
                if transcription:
                    await websocket.send_json({
                        "transcription": transcription,
                        "score": score,
                        "name": name,
                        "link": link,
                    })
                transcribe_fragments.clear()
                check_fragments.clear()
                total_duration_ms = 0
                check_duration_ms = 0
                triggered = False
                silence_duration_ms = 0
                print("Буфер очищен, режим транскрибации деактивирован.")
    
    except WebSocketDisconnect:
        if transcribe_fragments and triggered:
            transcription = transcribe_audio(transcribe_fragments)
            if transcription:
                print(f"Последняя транскрибация: {transcription}")
        check_fragments.clear()
        transcribe_fragments.clear()
        print("Клиент отключился")
    except Exception as e:
        print(f"Ошибка: {e}")