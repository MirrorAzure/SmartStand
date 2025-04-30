import asyncio
import websockets
import queue
import threading
import time
import os
import sys
from pydub import AudioSegment
import io

# Параметры эмуляции аудиопотока
CHUNK = 1024              # Размер фрагмента аудио (байт)
FRAME_RATE = 44100        # Частота дискретизации
SAMPLE_WIDTH = 2          # 16 бит = 2 байта на сэмпл
CHANNELS = 1              # Моно

def capture_audio_from_file(audio_queue, stop_event, file_path):
    """Эмуляция микрофона: чтение и конвертация аудиофайла по фрагментам"""
    try:
        # Загружаем аудиофайл
        audio = AudioSegment.from_file(file_path)
        # Конвертируем в нужный формат: 16-бит PCM, моно, 44100 Гц
        audio = audio.set_channels(CHANNELS).set_frame_rate(FRAME_RATE).set_sample_width(SAMPLE_WIDTH)
        
        print(f"Эмуляция микрофона начата для файла {file_path}. Для остановки нажмите Ctrl+C.")
        
        # Экспортируем в сырой PCM-поток
        raw_data = io.BytesIO()
        audio.export(raw_data, format="raw")
        raw_data.seek(0)
        
        # Рассчитываем задержку для имитации реального времени
        chunk_duration = CHUNK / (FRAME_RATE * SAMPLE_WIDTH * CHANNELS)
        
        while not stop_event.is_set():
            data = raw_data.read(CHUNK)
            if not data:
                break
            audio_queue.put(data)
            time.sleep(chunk_duration)  # Задержка для имитации реального времени
        
        print("Эмуляция микрофона завершена: файл полностью прочитан.")
        
    except Exception as e:
        print(f"Ошибка при чтении или конвертации аудиофайла: {e}")

async def send_audio(websocket, audio_queue):
    """Отправка аудиоданных из очереди на сервер"""
    while True:
        try:
            data = audio_queue.get_nowait()
            await websocket.send(data)
            #print(f"Отправлено {len(data)} байт")
        except queue.Empty:
            await asyncio.sleep(0.01)  # Небольшая задержка, чтобы не нагружать CPU

async def receive_responses(websocket):
    """Прием ответов от сервера"""
    while True:
        try:
            response = await websocket.recv()
            print(f"Получен ответ от сервера: {response}")
        except websockets.exceptions.ConnectionClosed:
            print("Соединение закрыто сервером")
            break

async def main(file_path):
    uri = "ws://localhost:8000/ws"  # Адрес вашего WebSocket-сервера
    audio_queue = queue.Queue()     # Очередь для передачи аудиоданных
    stop_event = threading.Event()  # Событие для остановки эмуляции
    
    # Запуск эмуляции микрофона в отдельном потоке
    capture_thread = threading.Thread(target=capture_audio_from_file, 
                                   args=(audio_queue, stop_event, file_path))
    capture_thread.start()
    
    try:
        async with websockets.connect(uri) as websocket:
            # Запуск задачи для отправки аудио
            send_task = asyncio.create_task(send_audio(websocket, audio_queue))
            # Запуск задачи для приема ответов
            receive_task = asyncio.create_task(receive_responses(websocket))
            
            # Ожидание завершения отправки
            await send_task
            # Даем немного времени для приема последних ответов
            await asyncio.sleep(1)
            # Отменяем задачу приема
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        stop_event.set()  # Остановка эмуляции
        capture_thread.join()

if __name__ == "__main__":
    print(__file__)
    file_path = "data/voice_sample.wav"  # Укажите путь к вашему аудиофайлу
    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
    else:
        try:
            asyncio.run(main(file_path))
        except KeyboardInterrupt:
            print("\nПрограмма остановлена пользователем")
            sys.exit(0)