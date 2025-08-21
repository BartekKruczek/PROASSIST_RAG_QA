import gc
import os

import torch
from llama_cpp import Llama

MODEL_DIR = os.getenv("HF_HOME", "./cache")
os.environ["HF_HOME"] = MODEL_DIR


def _download_model_from_hf_hub(
    model_id: str,
    model_filename: str,
    download_path: str = MODEL_DIR,
) -> None:
    """
    Download a model from Hugging Face Hub in GGUF format and returns the local path.
    """
    tmp = Llama.from_pretrained(
        repo_id=model_id,
        local_dir=download_path,
        filename=model_filename,
    )
    print(
        f"Model {model_id} downloaded and saved to {os.path.join(download_path, model_filename)}"
    )
    del tmp
    gc.collect()
    torch.cuda.empty_cache()


def main():
    """
    Execute function to download models from Hugging Face Hub.
    """
    _download_model_from_hf_hub(
        model_id="Qwen/Qwen3-4B-GGUF",
        model_filename="Qwen3-4B-Q4_K_M.gguf",
    )

    # Embeddings model
    _download_model_from_hf_hub(
        model_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
        model_filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
    )

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    print("Models downloaded successfully.")
