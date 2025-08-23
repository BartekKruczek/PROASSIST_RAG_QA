import itertools
import os
from pathlib import Path

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter


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


def create_texts_splitters() -> list[Document]:
    """
    Create a list of text splitters based on the content of files in the 'data' directory.

    Returns:
        list[Document]: A list containing lists of strings, where each inner list represents
                          the content of a file split into chunks.
    """
    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers)

    chunk_size = 1000
    chunk_overlap = 100
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    tmp_list: list[list[Document]] = []
    flatten_doc_list: list[Document] = []

    project_root = Path(__file__).resolve().parents[2]
    folder_path = project_root / "docs"

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".md"):
            file_path = folder_path / file_name
            text_content = _get_text_from_file(file_path)
            markdown_splitted = markdown_splitter.split_text(text_content)
            recursive_splitted = recursive_text_splitter.split_documents(
                markdown_splitted
            )

            # add file name to each document
            for doc in recursive_splitted:
                doc.metadata["source"] = file_name
                tmp_list.append(recursive_splitted)

    # flatten the list of lists into a single list
    flatten_doc_list = list(itertools.chain.from_iterable(tmp_list))

    return flatten_doc_list
