from langchain.schema import Document
from langchain_community.vectorstores import FAISS


def create_vector_db(
    flat_doc_list: list[Document],
    embeddings: object,
) -> FAISS:
    """
    Create a FAISS vector database for storing embeddings.

    Args:
        flat_doc_list (list[Document]): A list of Document objects containing text chunks.
        embeddings (object): An instance of an embeddings model to convert text to vectors.

    Returns:
        FAISS: An instance of the FAISS vector store.
    """
    vector_db = FAISS.from_documents(
        flat_doc_list, embeddings, distance_strategy="COSINE"
    )

    return vector_db
