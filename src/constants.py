# Параметры аудио
INPUT_SAMPLE_RATE = 44100  # Исходная частота дискретизации (Гц)
OUTPUT_SAMPLE_RATE = 16000  # Частота для Faster-Whisper и VAD (Гц)
CHANNELS = 1               # Моно
SAMPLE_WIDTH = 2           # 16 бит = 2 байта на сэмпл
CHECK_INTERVAL_MS = 1000   # Интервал проверки ключевого слова (мс)
PAUSE_DURATION_MS = 1000   # Длительность паузы для завершения транскрибации (мс)
VAD_FRAME_MS = 30          # Длительность кадра для VAD (10, 20 или 30 мс)

KEYWORD = "шокин"