import logging
import torch
import gc
from typing import List

from agents.common.interfaces import EmbeddingModel
from agents.common.enums import TextEmbeddingModel

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

    def embed(self, text: str, is_query: bool = False) -> List[float]:
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

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        if not self.model:
            self.load()
        
        input_text = text
        if is_query:
            input_text = "Represent this sentence for searching relevant passages: " + text
            
        embedding = self.model.encode(input_text, normalize_embeddings=True)
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

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        if not self.model: self.load()
        
        prefix = "search_document: "
        if is_query:
            prefix = "search_query: "
            
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

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        if not self.model: self.load()
        # BGE-M3 doesn't strictly require instructions for dense retrieval in the same way as v1.5, 
        # but consistency is good. However, standard usage often omits it for M3 dense.
        # We'll stick to raw text for M3 unless specified otherwise.
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
        model_name = config.get("embedding_model", TextEmbeddingModel.MINILM_L6.value)
        
        if model_name == TextEmbeddingModel.BAAI_BGE_LARGE.value:
            return BGEAdapter(model_name=model_name)
        elif model_name == TextEmbeddingModel.NOMIC_EMBED_V1_5.value:
            return NomicEmbedAdapter(model_name=model_name)
        elif model_name == TextEmbeddingModel.BGE_M3.value:
            return BGEM3Adapter(model_name=model_name)
        elif model_name == TextEmbeddingModel.MINILM_L6.value:
            return SentenceTransformerAdapter(model_name=model_name)
        else:
            # Fallback or custom model name
            return SentenceTransformerAdapter(model_name=model_name)
