import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
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
    ).as_retriever(search_kwargs={"k": 2})

    # check what retriever returns
    print("Retrieving documents for the question...")
    docs = retriever.invoke(args.question)
    for doc in docs:
        print(f"Retrieved document: {doc.page_content[:100]}...", doc.metadata)

    system_prompt = (
        "Jesteś asystentem QA. </no_think> "
        "Odpowiadaj WYŁĄCZNIE jednym krótkim zdaniem. </no_think> "
        "Jeśli nie znasz odpowiedzi, napisz: 'Nie wiem'. </no_think> "
        "Nie pokazuj swojego rozumowania. </no_think> "
        "Odpowiadaj wyłącznie gotową odpowiedzią. </no_think> "
        "Odpowiadaj zawsze po polsku.\n\n </no_think> "
        "Kontekst: {input_documents}\n\n </no_think> "
    )

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input_documents}\n\nPytanie: {question}"),
        ]
    )
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Podsumuj wszystkie odpowiedzi w jedno krótkie zdanie. </no_think>",
            ),
            ("human", "{summaries}"),
        ]
    )

    map_chain = LLMChain(llm=chat_llm, prompt=map_prompt, verbose=True)
    reduce_llm_chain = LLMChain(llm=chat_llm, prompt=reduce_prompt, verbose=True)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain, document_variable_name="summaries", verbose=True
    )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=200,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="input_documents",
        verbose=True,
    )

    print("RAG_QA application is ready to answer questions.")

    question = args.question
    retrieved_docs = retriever.invoke(question)

    response = map_reduce_chain.invoke(
        {"input_documents": retrieved_docs, "question": question},
        config={"callbacks": [ConsoleCallbackHandler()]},
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
