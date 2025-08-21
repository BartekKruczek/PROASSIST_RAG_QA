import argparse
import json
import os

from llama_cpp import Llama

MODEL_DIR = os.getenv("HF_HOME", "./cache")
os.environ["HF_HOME"] = MODEL_DIR


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Create and return an argument parser for downloading models from Hugging Face Hub.
    """
    parser = argparse.ArgumentParser(
        description="Download models from Hugging Face Hub."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help="Directory to save the downloaded models.",
    )
    parser.add_argument(
        "--chat-model-id",
        type=str,
        default="Qwen/Qwen3-14B-GGUF",
        help="Hugging Face model ID for the chat model.",
    )
    parser.add_argument(
        "--chat-model-filename",
        type=str,
        default="Qwen3-14B-Q6_K.gguf",
        help="Filename for the chat model.",
    )
    parser.add_argument(
        "--embedding-model-id",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B-GGUF",
        help="Hugging Face model ID for the embedding model.",
    )
    parser.add_argument(
        "--embedding-model-filename",
        type=str,
        default="Qwen3-Embedding-0.6B-f16.gguf",
        help="Filename for the embedding model.",
    )
    parser.add_argument(
        "--kwargs",
        type=json.loads,
        default={},
        help="Additional keyword arguments for the model download.",
    )

    args = parser.parse_args()

    return args


def _download_model_from_hf_hub(
    model_id: str,
    model_filename: str,
    download_path: str = MODEL_DIR,
    **kwargs: dict,
) -> None:
    """
    Download a model from Hugging Face Hub in GGUF format and returns the local path.

    Args:
        model_id (str): The Hugging Face model ID to download.
        model_filename (str): The filename to save the downloaded model.
        download_path (str): The directory where the model will be saved.
        **kwargs: Additional keyword arguments for Llama.from_pretrained.

    Returns:
        None: The function saves the model to the specified directory.
    """
    _ = Llama.from_pretrained(
        repo_id=model_id,
        local_dir=download_path,
        filename=model_filename,
        n_gpu_layers=-1,
        use_mlock=True,
        n_ctx=40960,
        **kwargs,
    )
    print(
        f"Model {model_id} downloaded and saved to {os.path.join(download_path, model_filename)}"
    )


def main():
    """
    Execute function to download models from Hugging Face Hub.
    """
    args = get_argument_parser()

    os.makedirs(args.model_dir, exist_ok=True)

    _download_model_from_hf_hub(
        model_id=args.embedding_model_id,
        model_filename=args.embedding_model_filename,
        download_path=args.model_dir,
        **args.kwargs,
    )

    _download_model_from_hf_hub(
        model_id=args.chat_model_id,
        model_filename=args.chat_model_filename,
        download_path=args.model_dir,
        **args.kwargs,
    )


if __name__ == "__main__":
    main()
    print("Models downloaded successfully.")
