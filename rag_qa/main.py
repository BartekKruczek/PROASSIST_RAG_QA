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


def get_gpu_config() -> dict:
    """
    Return the configuration for the chat model if running on GPU.
    """
    cnfg = {
        "n_gpu_layers": -1,
        "flash_attn": True,
        "use_mlock": True,
    }
    return cnfg


def main():
    """
    Execute the rag_qa pipeline.
    """
    set_verbose(True)
    set_debug(True)
    cnfg = get_gpu_config()

    chat_llm = load_chat_model(
        max_tokens=2048,
        n_ctx=40960,
        verbose=False,
        n_gpu_layers=cnfg["n_gpu_layers"],
        use_mlock=cnfg["use_mlock"],
    )

    print("Warming up the chat model...")
    response = chat_llm.invoke("What is the capital of Poland?")
    print(response)

    texts, file_names = create_texts_splitters(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0"
    )

    retriever = create_vector_db(
        texts=texts,
        file_names=file_names,
        embeddings=load_embeddings_model(
            model_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
            model_filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
            n_gpu_layers=cnfg["n_gpu_layers"],
            use_mlock=cnfg["use_mlock"],
            n_ctx=32768,
            verbose=False,
        ),
    ).as_retriever(search_kwargs={"k": 1})

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat_llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    print("RAG_QA application is ready to answer questions.")

    question = "Kto stworzył PLLuM?"
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
