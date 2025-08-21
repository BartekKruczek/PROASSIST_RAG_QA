import gc
import os

import torch
from llama_cpp import Llama

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = "./cache"


def _download_model_from_hf_hub(
    model_id: str = "Qwen/Qwen3-4B-GGUF",
    download_path: str = os.getenv("HF_HOME"),
    model_filename: str = "Qwen3-4B-Q4_K_M.gguf",
) -> None:
    """
    Download a model from Hugging Face Hub in GGUF format and returns the local path.

    Args:
        model_id (str): The ID of the model on Hugging Face Hub.
        download_path (str): The local path to save the downloaded model.
        model_filename (str): The name of the file to save the model as.

    Returns:
        None: The model is loaded into memory.
    """
    tmp = Llama.from_pretrained(
        repo_id=model_id,
        local_dir=download_path,
        filename=model_filename,
    )

    print(f"Model {model_id} downloaded and saved to {download_path}/{model_filename}")
    del tmp
    gc.collect()
    torch.cuda.empty_cache()


def main():
    """
    Download selected model from Hugging Face Hub and save it to the specified path.
    """
    # chat model parameters
    model_id = "Qwen/Qwen3-4B-GGUF"
    download_path = os.getenv("HF_HOME", "./models")
    model_filename = "Qwen3-4B-Q4_K_M.gguf"

    os.makedirs(download_path, exist_ok=True)
    _download_model_from_hf_hub(model_id, download_path, model_filename)

    # embeddings model parameters
    model_id: str = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    download_path: str = os.getenv("HF_HOME", "./models")
    model_filename: str = "Qwen3-Embedding-0.6B-Q8_0.gguf"

    os.makedirs(download_path, exist_ok=True)
    _download_model_from_hf_hub(model_id, download_path, model_filename)

    # free up memory
    gc.collect()
    torch.cuda.empty_cache()
