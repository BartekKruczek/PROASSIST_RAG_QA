from langchain.schema import Document
from langchain.vectorstores import FAISS


def create_vector_db(
    texts: list[list[str]],
    file_names: list[str],
    embeddings: object,
) -> FAISS:
    """
    Create a FAISS vector database for storing embeddings.

    Args:
        texts (list[list[str]]): A list of lists containing text chunks.
        file_names (list[str]): A list of file names corresponding to the text chunks.
        embeddings (object): An instance of an embeddings model to convert text to vectors.

    Returns:
        FAISS: An instance of the FAISS vector store.
    """
    documents = []
    for file_chunks, file_name in zip(texts, file_names, strict=False):
        documents.extend(
            Document(page_content=chunk, metadata={"source": file_name})
            for chunk in file_chunks
        )

    vector_db = FAISS.from_documents(documents, embeddings)

    return vector_db
