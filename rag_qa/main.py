import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.globals import set_debug, set_verbose
from langchain_core.prompts import ChatPromptTemplate

from logger.create_logger import create_logger
from utils.embeddings.faiss_storage import create_vector_db
from utils.model_loader.loader import (
    load_chat_model,
    load_embeddings_model,
)
from utils.text_splitter.splitter import create_texts_splitters


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Create and return an argument parser for the RAG_QA application.
    """
    parser = argparse.ArgumentParser(description="RAG_QA Application")

    parser.add_argument(
        "--chat-model-filename",
        type=str,
        default="Qwen3-14B-Q6_K.gguf",
        help="Filename for the chat model.",
    )
    parser.add_argument(
        "--embedding-model-filename",
        type=str,
        default="Qwen3-Embedding-0.6B-f16.gguf",
        help="Filename for the embedding model.",
    )
    parser.add_argument(
        "--text-splitter-model-id",
        type=str,
        default="Snowflake/snowflake-arctic-embed-l-v2.0",
        help="Hugging Face model ID for the text splitter.",
    )
    parser.add_argument(
        "--chat-model-kwargs",
        type=json.loads,
        default={},
        help="Additional keyword arguments for the chat model.",
    )
    parser.add_argument(
        "--embedding-model-kwargs",
        type=json.loads,
        default={},
        help="Additional keyword arguments for the embedding model.",
    )
    parser.add_argument(
        "--text-splitter-kwargs",
        type=json.loads,
        default={},
        help="Additional keyword arguments for the text splitter.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The question to ask the RAG_QA application.",
    )

    return parser.parse_args()


def main():
    """
    Execute the rag_qa pipeline.
    """
    set_verbose(True)
    set_debug(True)

    args = get_argument_parser()

    chat_llm = load_chat_model(
        model_filename=args.chat_model_filename,
        temperature=0.01,
        top_p=0.95,
        repeat_penalty=1.1,
        max_tokens=128,
        verbose=False,
        n_gpu_layers=-1,
        use_mlock=True,
        n_ctx=32768,
        yarn_orig_ctx=32768,
        rope_scaling_type=2,
        yarn_attn_factor=4.0,
        model_kwargs={
            "chat_format": "chatml",
            **args.chat_model_kwargs,
        },
    )

    print("Warming up the chat model...")
    response = chat_llm.invoke("Co jest stolicą Polski? /no_think")
    print(response)

    texts, file_names = create_texts_splitters(
        model_name=args.text_splitter_model_id,
        **args.text_splitter_kwargs,
    )

    retriever = create_vector_db(
        texts=texts,
        file_names=file_names,
        embeddings=load_embeddings_model(
            model_filename=args.embedding_model_filename,
            n_gpu_layers=-1,
            use_mlock=True,
            n_ctx=32768,
            verbose=False,
            **args.embedding_model_kwargs,
        ),
    ).as_retriever(search_kwargs={"k": 1})

    # check what retriever returns
    print("Retrieving documents for the question...")
    docs = retriever.get_relevant_documents(args.question)
    for doc in docs:
        print(f"Retrieved document: {doc.page_content[:100]}...", doc.metadata)

    system_prompt = (
        "Jesteś asystentem QA. "
        "Odpowiadaj WYŁĄCZNIE jednym krótkim zdaniem. "
        "Jeśli nie znasz odpowiedzi, napisz: 'Nie wiem'. "
        "Nie pokazuj swojego rozumowania. "
        "Odpowiadaj wyłącznie gotową odpowiedzią. "
        "Odpowiadaj zawsze po polsku.\n\n"
        "Kontekst: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        chat_llm, prompt, document_variable_name="context"
    )
    chain = create_retrieval_chain(retriever, question_answer_chain)

    print("RAG_QA application is ready to answer questions.")

    question = args.question
    response = chain.invoke(
        {"input": question}, config={"callbacks": [ConsoleCallbackHandler()]}
    )
    print(f"Question: {question}")
    print(f"Response: {response}")


if __name__ == "__main__":
    try:
        print("Starting the RAG_QA application...")
        main()
        print("RAG_QA application executed successfully.")
    except Exception:
        logger = create_logger("rag_qa")
        logger.exception("An error occurred in the rag_qa application.")
        raise
