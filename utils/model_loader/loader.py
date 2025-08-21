import os

from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)


def load_chat_model(
    model_filename: str = "Qwen3-4B-Q4_K_M.gguf",
    **kwargs,
) -> LlamaCpp:
    """
    Load the chat model from local cache.

    Args:
        model_filename (str): The name of the file to save the model as.
        **kwargs: Additional keyword arguments for the LlamaCpp model.

    Returns:
        LlamaCpp: An instance of the LlamaCpp model.
    """
    llm = LlamaCpp(
        model_path=os.getenv("HF_HOME", "./models") + "/" + model_filename,
        **kwargs,
    )
    return llm


def load_embeddings_model(
    model_filename: str = "Qwen3-Embedding-0.6B-Q8_0.gguf",
    **kwargs,
) -> LlamaCppEmbeddings:
    """
    Load the embeddings model from local cache.

    Args:
        model_filename (str): The name of the file to save the model as.
        **kwargs: Additional keyword arguments for downloading the embeddings model.

    Returns:
        LlamaCppEmbeddings: An instance of the Llama embeddings model.
    """
    llm_embeddings = LlamaCppEmbeddings(
        model_path=os.getenv("HF_HOME", "./models") + "/" + model_filename,
        **kwargs,
    )
    return llm_embeddings


def load_sentence_transformers_model(**kwargs) -> SentenceTransformersTokenTextSplitter:
    """
    Load the sentence transformers model from local cache.

    Args:
        **kwargs: Additional keyword arguments for the sentence transformers model.

    Returns:
        SentenceTransformersTokenTextSplitter: An instance of the sentence transformers model.
    """
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=1024,
        chunk_overlap=50,
        **kwargs,
    )
    return text_splitter
