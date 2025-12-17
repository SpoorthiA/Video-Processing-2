import logging
import torch
import gc
from typing import List

from agents.common.interfaces import EmbeddingModel

logger = logging.getLogger(__name__)

class SentenceTransformerAdapter(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if not self.model:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str) -> List[float]:
        if not self.model:
            self.load()
        
        # SentenceTransformer returns numpy array, convert to list
        embedding = self.model.encode(text)
        return embedding.tolist()

    def get_dimension(self) -> int:
        if not self.model:
            self.load()
        return self.model.get_sentence_embedding_dimension()

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class BGEAdapter(EmbeddingModel):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if not self.model:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading BGE Model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str) -> List[float]:
        if not self.model:
            self.load()
        
        # BGE often requires specific instruction for queries, but for general embedding:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        if not self.model:
            self.load()
        return self.model.get_sentence_embedding_dimension()

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class NomicEmbedAdapter(EmbeddingModel):
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if not self.model:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading Nomic Embed: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)

    def embed(self, text: str) -> List[float]:
        if not self.model: self.load()
        
        # Nomic requires prefix for tasks. Assuming document embedding for indexing.
        prefix = "search_document: "
        embedding = self.model.encode(prefix + text)
        return embedding.tolist()

    def get_dimension(self) -> int:
        if not self.model: self.load()
        return self.model.get_sentence_embedding_dimension()

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

class BGEM3Adapter(EmbeddingModel):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if not self.model:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading BGE-M3: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str) -> List[float]:
        if not self.model: self.load()
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        if not self.model: self.load()
        return self.model.get_sentence_embedding_dimension()

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

class EmbeddingFactory:
    @staticmethod
    def get_model(config: dict) -> EmbeddingModel:
        model_name = config.get("embedding_model", "all-minilm-l6-v2")
        
        if model_name == "bge-large-en-v1.5":
            return BGEAdapter(model_name="BAAI/bge-large-en-v1.5")
        elif model_name == "nomic-embed":
            return NomicEmbedAdapter()
        elif model_name == "bge-m3":
            return BGEM3Adapter()
        elif model_name == "all-minilm-l6-v2":
            return SentenceTransformerAdapter(model_name="all-MiniLM-L6-v2")
        else:
            # Fallback or custom model name
            return SentenceTransformerAdapter(model_name=model_name)
