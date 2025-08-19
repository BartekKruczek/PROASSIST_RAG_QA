import os

from utils.model_loader.loader import load_sentence_transformers_model


def _get_text_from_file(file_path: str) -> str:
    """
    Read the content of a file and returns it as a string.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of the file.
    """
    with open(file_path, encoding="utf-8") as file:
        return file.read()


def create_texts_splitters(**kwargs) -> list[list[str]]:
    """
    Create a list of text splitters based on the content of files in the 'data' directory.

    Args:
        **kwargs: Additional keyword arguments for the text splitter.

    Returns:
        list[list[str]]: A list containing lists of strings, where each inner list represents
                          the content of a file split into chunks.
    """
    model_splitter = load_sentence_transformers_model(**kwargs)
    folder_path: str = "./docs/"
    texts_splitters = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            text_content = _get_text_from_file(file_path)
            text_splitter = model_splitter.split_text(text_content)
            texts_splitters.append(text_splitter)

    return texts_splitters
