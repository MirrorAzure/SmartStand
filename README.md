# Smart Stand

Бэкэнд для умного стенда с функцией отклика на ключевое слово и возвращение наиболее релевантной ссылки.

## Запуск
### Docker
```bash
docker compose up
```

## Переменные среды
Основные настройки системы проводятся с помощью файла .env
### .env
```env
DEVICE=cpu # cpu, cuda

WHISPER_MODEL_SIZE=tiny # tiny, base, small, large-v3, large-v3-turbo
COMPUTE_TYPE=int8 # int8, float16

VECTOR_COLLECTION_NAME=links
QDRANT_HOST=localhost
QDRANT_PORT=6333
```
`DEVICE` - устройство (`cpu` или `cuda`);   
`WHISPER_MODEL_SIZE` - размер модели ASR, для инференса на CPU рекомендуется использовать модели меньшего размера;  
`COMPUTE_TYPE` - квантование модели, при инференсе на CPU доступно только `int8`, для  GPU рекомендуется использовать `float16`;  
`VECTOR_COLLECTION_NAME` - название коллекции ссылок в qdrant;  
`QDRANT_HOST` - хост qdrant;  
`QDRANT_PORT` - порт qdrant.

## Входные данные
Система принимает соединение по WebSocket на порту `8000`.  
Сервер ждёт поток байт, представляющих собой 16-бит PCM, моно, 44100 Гц.  
Пример кода для фронтенда:
```js
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const audioContext = new AudioContext({ sampleRate: 44100 });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(1024, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);
    processor.onaudioprocess = e => {
      const inputData = e.inputBuffer.getChannelData(0); // Float32 массив
      // Конвертация Float32 (-1.0 до 1.0) в Int16 (-32768 до 32767)
      const pcmData = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
      }
      const buffer = pcmData.buffer; // ArrayBuffer
      // Отправка через WebSocket
      websocket.send(buffer);
    };
  });
```

Также системе для работы необходим список ссылок на ресурсы. Он задаётся в файле `data/links.json` и должен иметь следующую структуру:
```json
[
    {
        "name": "Название ссылки",
        "link": "Адрес ссылки"
    },
    ...
]
```

## Выходные данные
При обнаружении ключевого слово сервер передаёт через WebSocket JSON-объект со следующей структурой:
```json
{
    "transcription": "Результат транскрибации",
    "score": 0.25, // Уверенность модели
    "name": "Имя ссылки",
    "link": "Ссылка на самый релевантный ресурс",
}
```