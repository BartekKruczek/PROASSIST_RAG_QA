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
