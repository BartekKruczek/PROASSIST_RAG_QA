import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from langchain_community.llms.llamacpp import LlamaCpp
from llama_cpp import Llama


def _download_model_from_hf_hub(
    model_id: str = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    download_path: str = "./cache/llama_cpp",
    model_filename: str = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
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


def load_chat_model(**kwargs) -> LlamaCpp:
    """
    Load the chat model from local cache.

    Args:
        **kwargs: Additional keyword arguments for the Llama model.

    Returns:
        Llama: An instance of the Llama model.
    """
    download_path: str = _download_model_from_hf_hub()

    llm = LlamaCpp(
        model_path=download_path,
        **kwargs,
    )

    return llm
