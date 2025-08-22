# PROASSIST_RAG_QA

This repository contains a Python implementation of a Retrieval-Augmented Generation (RAG) system using LangChain, FAISS, Hugging Face Transformers and Sentence Transformers. The system is designed to answer questions based on a set of .md files, which are processed and indexed for efficient retrieval.

## How to use

1. **Clone the repository**: In order to start, one should download created repository:

    ```bash
    git clone https://github.com/BartekKruczek/PROASSIST_RAG_QA.git
    ```

2. **Add repository to path**: After cloning, you need to add the repository to your Python path. This could be done by following CLI commands:

    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```

3. **Install dependencies**: Ensure you have the required Python packages installed. You can install them using [uv](https://docs.astral.sh/uv/)
   from the root directory of the project:

   ```bash
   uv sync
   ```

4. **Download models**: 

    Before running the main scipt, you need to locally download selected models. What is more, they should be in [GGUF](https://huggingface.co/docs/hub/gguf) format to fully utilize the [llama.cpp](https://github.com/abetlen/llama-cpp-python) library. You can download them using the following commands:

    ```bash
    uv run rag_qa/hf_download.py
    ```

5. **Run main script**: After successfully downloading models, you can run the main script to start the RAG pipeline:

    ```bash
    uv run rag_qa/rag_main.py
    ```

## Argparse - existing options
It is worth to mention that there are some CLI options available for the `hf_download.py` and `rag_main.py` scripts as follows:

1. **hf_download.py**:
   - `--model-dir`: Directory where the models will be downloaded.
   - `--chat-model-id`: Name of the chat model to download from Hugging Face in the GGUF format.
   - `--chat-model-filename`: Name of the file to save the quantized chat model.
   - `--embedding-model-id`: Name of the embedding model to download from Hugging Face in the GGUF format.
   - `--embedding-model-filename`: Name of the file to save the quantized embedding model.
   - `--kwargs`: Additional keyword arguments for the `from_pretrained` method of the model. It should be provided in JSON format, e.g. `{"n_ctx": 2048}`.

2. **rag_main.py**:
   - `--chat-model-filename`: Name of the chat model file to use for generating responses.
   - `--embedding-model-filename`: Name of the embedding model file to use for generating embeddings.
   - `--text-splitter-model-id`: Name of the text splitter model to download from Hugging Face.
   - `--chat-model-kwargs`: Additional keyword arguments for the chat model, provided in JSON format.
   - `--embedding-model-kwargs`: Additional keyword arguments for the embedding model, provided in JSON format.
   - `--text-splitter-kwargs`: Additional keyword arguments for the text splitter, provided in JSON format.
  
For example, to download different model that is set to default, you can run:

```bash
uv run rag_qa/hf_download.py --chat-model-id Qwen/Qwen3-14B-GGUF --chat-model-filename Qwen3-14B-Q8_0.gguf
```

Then you have to pass the same model filename to the `rag_main.py` script:

```bash
uv run rag_qa/main.py --chat-model-filename Qwen3-14B-Q8_0.gguf
```

## Used models

In this section, we dive into the specifics of used models. We can divide them into three main categories: chat model, embedding model and text splitter model. Each of them can be chosen from a variety of models available on Hugging Face. Below, only the default models are mentioned.

### Chat model

For inference purposes, the [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B-GGUF) model is used. Not only it was trained on polish language dataset, but also it leverage in human preference alignment.

### Hardware acceleration

It is **HARDLY** recommended to use any kind of hardware acceleration, such as GPU or Metal. 

## Asked questions

Before starting RAG pipeline, simple warmup question is being provided to the chat model: `What is the capital of Poland?`. This is to ensure that the model is loaded and ready to process further queries.

## Results

## Future improvements
- Add CLI parser for user input. It would give more flexibility and control over workflow.

## Summary

## References