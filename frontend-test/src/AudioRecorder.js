import React, { useEffect, useState } from 'react';
import { w3cwebsocket as W3CWebSocket } from 'websocket';

const AudioRecorder = () => {
    const [ws, setWs] = useState(null);
    const [texts, setTexts] = useState([]);  // Храним список текстов
    const [dots, setDots] = useState(1);

    useEffect(() => {
        const websocket = new W3CWebSocket('ws://localhost:8000/ws');
        websocket.onopen = () => {
            console.log('WebSocket connection opened');
        };
        websocket.onmessage = (message) => {
            const data = JSON.parse(message.data);
            if (data.status === 1) {
                console.log(data);
                setTexts(prev => [...prev, data.transcript]);  // TODO: можно избавиться от этой строчки
                window.open(data.payload.link, "_blank");  // Для открытие в той же вкладке использовать "_self"
            }
        };
        websocket.onclose = () => {
            console.log('WebSocket connection closed');
        };
        setWs(websocket);

        return () => {
            websocket.close();
        };
    }, []);

    useEffect(() => {
        if (ws) {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    const audioContext = new AudioContext({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(stream);
                    const processor = audioContext.createScriptProcessor(2048, 1, 1);

                    source.connect(processor);
                    processor.connect(audioContext.destination);

                    processor.onaudioprocess = (e) => {
                        const audioData = e.inputBuffer.getChannelData(0);
                        const int16Data = convertFloat32ToInt16(audioData);
                        ws.send(int16Data);
                    };
                })
                .catch(err => console.error('Error accessing microphone:', err));
        }
    }, [ws]);

    useEffect(() => {
        const interval = setInterval(() => {
            setDots(prev => (prev % 3) + 1);
        }, 500);
        return () => clearInterval(interval);
    }, []);

    const convertFloat32ToInt16 = (buffer) => {
        const l = buffer.length;
        const buf = new Int16Array(l);
        for (let i = 0; i < l; i++) {
            buf[i] = Math.min(1, buffer[i]) * 0x7FFF;
        }
        return buf.buffer;
    };

    return (
        <div style={{ position: 'fixed', bottom: '20px', right: '20px', display: 'flex', flexDirection: 'column', background: 'rgba(0, 0, 0, 0.5)', padding: '10px', borderRadius: '10px', color: 'white', animation: 'levitate 2s ease-in-out infinite' }}>
            <style>
                {`
                    @keyframes levitate {
                        0% { transform: translateY(0); }
                        50% { transform: translateY(-10px); }
                        100% { transform: translateY(0); }
                    }
                `}
            </style>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ marginRight: '10px' }}>🎙️</div>
                <div style={{ position: 'relative', width: 'auto' }}>
                    <span style={{ visibility: 'hidden', whiteSpace: 'nowrap' }}>Шокин слушает...</span>
                    <span style={{ position: 'absolute', left: 0, top: 0, whiteSpace: 'nowrap' }}>Шокин слушает{'.'.repeat(dots)}</span>
                </div>
            </div>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, maxHeight: '200px', overflowY: 'auto' }}>
                {texts.map((text, index) => (
                    <li key={index} style={{ marginBottom: '5px' }}>{text}</li>
                ))}
            </ul>
        </div>
    );
};

export default AudioRecorder;