import os

from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)

MODEL_DIR = os.getenv("HF_HOME", "./cache")
os.environ["HF_HOME"] = MODEL_DIR


def load_chat_model(
    model_filename: str = "Qwen3-4B-Q4_K_M.gguf",
    **kwargs,
) -> LlamaCpp:
    """
    Load a chat model from the specified filename in the MODEL_DIR.

    Args:
        model_filename (str): The name of the model file to load.
        **kwargs: Additional keyword arguments for LlamaCpp.

    Returns:
        LlamaCpp: An instance of the LlamaCpp model loaded from the specified file.
    """
    model_path = os.path.join(MODEL_DIR, model_filename)
    return LlamaCpp(model_path=model_path, **kwargs)


def load_embeddings_model(
    model_filename: str = "Qwen3-Embedding-0.6B-Q8_0.gguf",
    **kwargs,
) -> LlamaCppEmbeddings:
    """
    Load an embeddings model from the specified filename in the MODEL_DIR.

    Args:
        model_filename (str): The name of the model file to load.
        **kwargs: Additional keyword arguments for LlamaCppEmbeddings.

    Returns:
        LlamaCppEmbeddings: An instance of the LlamaCppEmbeddings model loaded from the specified file.
    """
    model_path = os.path.join(MODEL_DIR, model_filename)
    return LlamaCppEmbeddings(model_path=model_path, **kwargs)


def load_sentence_transformers_model(**kwargs) -> SentenceTransformersTokenTextSplitter:
    """
    Load a SentenceTransformersTokenTextSplitter with default parameters.

    Args:
        **kwargs: Additional keyword arguments for SentenceTransformersTokenTextSplitter.

    Returns:
        SentenceTransformersTokenTextSplitter: An instance of the text splitter with specified chunk size and overlap.
    """
    return SentenceTransformersTokenTextSplitter(
        chunk_size=1024,
        chunk_overlap=50,
        **kwargs,
    )
