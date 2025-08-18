from logger.create_logger import create_logger


def main():
    """
    Main function to execute the RAG_QA application.
    """
    print("Hello World!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = create_logger("RAG_QA")
        logger.exception(f"An error occurred: {e}")
        raise
