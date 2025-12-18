from abc import ABC, abstractmethod
from typing import List, Any

class VisionModel(ABC):
    @abstractmethod
    def load(self):
        """Loads the model into memory."""
        pass
    
    @abstractmethod
    def process_image(self, image: Any) -> dict:
        """
        Processes the image and returns a dictionary.
        Dict can contain:
        - "text": str (Caption)
        - "embedding": List[float] (Visual Embedding)
        """
        pass
    
    @abstractmethod
    def unload(self):
        """Unloads the model to free resources."""
        pass

class SpeechModel(ABC):
    @abstractmethod
    def load(self):
        """Loads the model into memory."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[dict]:
        """
        Transcribes the audio file.
        Returns a list of segments: [{"start": float, "end": float, "text": str}]
        """
        pass

    @abstractmethod
    def unload(self):
        """Unloads the model to free resources."""
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    def load(self):
        """Loads the model into memory."""
        pass
    
    @abstractmethod
    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """Generates an embedding vector for the text."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        pass

    @abstractmethod
    def unload(self):
        """Unloads the model to free resources."""
        pass
