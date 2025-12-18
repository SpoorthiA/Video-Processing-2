from enum import Enum

class VisionModel(str, Enum):
    BLIP_BASE = "Salesforce/blip-image-captioning-base"
    FLORENCE_2_LARGE = "microsoft/Florence-2-large"
    QWEN_2_5_VL = "Qwen/Qwen2.5-VL-7B-Instruct"
    SIGLIP = "google/siglip-so400m-patch14-384"
    CLIP = "openai/clip-vit-base-patch32"

class SpeechModel(str, Enum):
    FASTER_WHISPER_BASE = "base"
    FASTER_WHISPER_LARGE = "large-v3"
    DISTIL_WHISPER_LARGE_V3 = "distil-large-v3"

class TextEmbeddingModel(str, Enum):
    MINILM_L6 = "all-MiniLM-L6-v2"
    NOMIC_EMBED_V1_5 = "nomic-ai/nomic-embed-text-v1.5"
    BGE_M3 = "BAAI/bge-m3"
    BAAI_BGE_LARGE = "BAAI/bge-large-en-v1.5"
