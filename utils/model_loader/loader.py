import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = "./cache"

from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)
from llama_cpp import Llama


def _download_model_from_hf_hub(
    model_id: str = "Qwen/Qwen3-4B-GGUF",
    download_path: str = os.getenv("HF_HOME"),
    model_filename: str = "Qwen3-4B-Q4_K_M.gguf",
    **kwargs,
) -> str:
    """
    Download a model from Hugging Face Hub in GGUF format and returns the local path.

    Args:
        model_id (str): The ID of the model on Hugging Face Hub.
        download_path (str): The local path to save the downloaded model.
        model_filename (str): The name of the file to save the model as.
        **kwargs: Additional keyword arguments for the download function.

    Returns:
        str: The local path with model ID appended where it was downloaded.
    """
    Llama.from_pretrained(
        repo_id=model_id,
        local_dir=download_path,
        filename=model_filename,
        **kwargs,
    )

    return f"{download_path}/{model_filename}"


def load_chat_model(
    model_id: str = "Qwen/Qwen3-4B-GGUF",
    download_path: str = os.getenv("HF_HOME"),
    model_filename: str = "Qwen3-4B-Q4_K_M.gguf",
    **kwargs,
) -> Llama:
    """
    Load the chat model from local cache.

    Args:
        model_id (str): The ID of the model on Hugging Face Hub.
        download_path (str): The local path to save the downloaded model.
        model_filename (str): The name of the file to save the model as.
        **kwargs: Additional keyword arguments for the Llama model.

    Returns:
        Llama: An instance of the Llama model.
    """
    llm = LlamaCpp(
        repo_id=model_id,
        local_dir=download_path,
        filename=model_filename,
        **kwargs,
    )

    return llm


def load_embeddings_model(
    **kwargs,
) -> LlamaCppEmbeddings:
    """
    Load the embeddings model from local cache.

    Args:
        **kwargs: Additional keyword arguments for downloading the embeddings model.

    Returns:
        LlamaCppEmbeddings: An instance of the Llama embeddings model.
    """
    llm_embeddings = LlamaCppEmbeddings(
        model_path=_download_model_from_hf_hub(**kwargs),
    )

    return llm_embeddings


def load_sentence_transformers_model(**kwargs) -> SentenceTransformersTokenTextSplitter:
    """
    Load the sentence transformers model from local cache.

    Args:
        **kwargs: Additional keyword arguments for downloading the sentence transformers model.

    Returns:
        SentenceTransformersTokenTextSplitter: An instance of the sentence transformers model.
    """
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=1024,
        chunk_overlap=50,
        **kwargs,
    )

    return text_splitter
