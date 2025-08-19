from logger.create_logger import create_logger
from utils.model_loader.loader import load_chat_model


def main():
    """
    Execute the rag_qa pipeline.
    """
    print("Hello World!")
    load_chat_model()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger = create_logger("rag_qa")
        logger.exception("An error occurred in the rag_qa application.")
        raise
