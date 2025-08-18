from logger.create_logger import create_logger


def main():
    """
    Execute the rag_qa pipeline.
    """
    print("Hello World!")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger = create_logger("rag_qa")
        logger.exception("An error occurred in the rag_qa application.")
        raise
