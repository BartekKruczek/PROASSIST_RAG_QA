from langchain.globals import set_verbose

from logger.create_logger import create_logger
from utils.model_loader.loader import load_chat_model


def main():
    """
    Execute the rag_qa pipeline.
    """
    print("Hello World!")
    set_verbose(False)

    llm = load_chat_model(
        max_tokens=16384,
    )

    question = """
    Question: A rap battle between Stephen Colbert and John Oliver
    """
    ans = llm.invoke(question)
    print(f"Answer: {ans}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger = create_logger("rag_qa")
        logger.exception("An error occurred in the rag_qa application.")
        raise
