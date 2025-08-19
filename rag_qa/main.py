from langchain.globals import set_verbose

from logger.create_logger import create_logger
from utils.embeddings.faiss_storage import create_vector_db
from utils.model_loader.loader import (
    load_chat_model,
    load_embeddings_model,
)
from utils.text_splitter.splitter import create_texts_splitters


def main():
    """
    Execute the rag_qa pipeline.
    """
    set_verbose(False)

    load_chat_model(
        max_tokens=16384,
    )

    texts, file_names = create_texts_splitters(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0"
    )

    create_vector_db(
        texts=texts,
        file_names=file_names,
        embeddings=load_embeddings_model(
            model_id="Qwen/Qwen3-Embedding-8B-GGUF",
            model_filename="Qwen3-Embedding-8B-Q4_K_M.gguf",
        ),
    )


if __name__ == "__main__":
    try:
        print("Starting the RAG_QA application...")
        main()
        print("RAG_QA application executed successfully.")
    except Exception:
        logger = create_logger("rag_qa")
        logger.exception("An error occurred in the rag_qa application.")
        raise
