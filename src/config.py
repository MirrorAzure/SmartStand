import os
import torch
# from pyaudio import paInt16

cuda_available = torch.cuda.is_available()

class ASRConfig:
    
    _model_size = os.environ.get("WHISPER_MODEL_SIZE", "large-v3-turbo" if cuda_available else "tiny")
    _device = os.environ.get("DEVICE", "cuda" if cuda_available else "cpu")
    _compute_type = os.environ.get("COMPUTE_TYPE", "float16" if cuda_available else "int8")
    
    @classmethod
    def get_model_size(cls):
        return cls._model_size
    
    @classmethod
    def get_device(cls):
        return cls._device
    
    @classmethod
    def get_compute_type(cls):
        return cls._compute_type

class VectorConfig:
    
    _collection_name = os.environ.get("VECTOR_COLLECTION_NAME", "links")
    _qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    _qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
    
    @classmethod
    def get_collection_name(cls):
        return cls._collection_name
    
    @classmethod
    def get_qdrant_host(cls):
        return cls._qdrant_host
    
    @classmethod
    def get_qdrant_port(cls):
        return cls._qdrant_port
    
class EmbeddingConfig:
    
    _model_name = os.environ.get("EMBEDDING_MODEL", "LaBSE")
    _device = os.environ.get("DEVICE", "cuda" if cuda_available else "cpu")
    
    @classmethod
    def get_model_path(cls):
        return f"models/{cls._model_name}"
    
    @classmethod
    def get_device(cls):
        return cls._device