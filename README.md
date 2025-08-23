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
   - `--question`: Required field, the question to be answered by the RAG system.
   - `--verbose`: Default to True, if set to False, the script will not print detailed information about the RAG pipeline execution.
   - `--chat-model-filename`: Name of the chat model file to use for generating responses.
   - `--embedding-model-filename`: Name of the embedding model file to use for generating embeddings.
   - `--chat-model-kwargs`: Additional keyword arguments for the chat model, provided in JSON format.
   - `--embedding-model-kwargs`: Additional keyword arguments for the embedding model, provided in JSON format.
  
For example, to download different model that is set to default, you can run:

```bash
uv run rag_qa/hf_download.py --chat-model-id Qwen/Qwen3-14B-GGUF --chat-model-filename Qwen3-14B-Q8_0.gguf
```

Then you have to pass the same model filename to the `rag_main.py` script:

```bash
uv run rag_qa/main.py --chat-model-filename Qwen3-14B-Q8_0.gguf --question "Jakie modele LLaMa są dostępne? /no_think"
```

## Used models

In this section, we dive into the specifics of used models. We can divide them into three main categories: chat model, embedding model and text splitter model. Each of them can be chosen from a variety of models available on Hugging Face. Below, only the default models are mentioned.

### Chat model

For inference purposes, the [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B-GGUF) model is used. Not only it was trained on polish language dataset, but also it leverage in human preference alignment.

### Embedding model

Embedding model belongs to the same Qwen family as the chat model. However, this time [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) model is significantly smaller due to hardware limitations. To minimize possible quality losses, largest GGUF format is used, which is f16.

### Hardware acceleration

It is **HARDLY** recommended to use any kind of hardware acceleration, such as GPU or Metal. By default, loading models, speedup is selected and fully utilized. Specified libraries should automatically detect available hardware and install necessary dependencies. However, if you are using GPU or Metal and see that only CPU is being used, please check out [this](https://llama-cpp-python.readthedocs.io/en/latest/#installation) tutorial how to install llama-cpp-python with enabled backend support.

## Asked questions

Before starting RAG pipeline, simple warmup question is being provided to the chat model: `What is the capital of Poland?`. This is to ensure that the model is loaded and ready to process further queries.

With given question, context was also proved. It helps the model to keep the answer short and relevant.
```python
"Jesteś asystentem QA. </no_think> "
"Odpowiadaj WYŁĄCZNIE jednym krótkim zdaniem. </no_think> "
"Jeśli nie znasz odpowiedzi, napisz: 'Nie wiem'. </no_think> "
"Nie pokazuj swojego rozumowania. </no_think> "
"Odpowiadaj wyłącznie gotową odpowiedzią. </no_think> "
"Odpowiadaj zawsze po polsku.\n\n </no_think> "
"Kontekst: {input_documents}\n\n </no_think> "
```

In total, 5 questions were asked:
- Jakie modele LLaMa są dostępne?
- Kto stworzył PLLuM?
- Jaki model najlepiej działa na GPU z 24 GB VRAM?
- Kiedy powstał pierwszy model typu GPT?
- Jakie są wymaganie sprzętowe dla modelu Llama 4 Scout?

## Results

In this section answers with context are provided for each question. In addition, the full output of the RAG pipeline is shown in the details section.

- Q: Jakie modele LLaMa są dostępne? \
  A: Dostępne są modele LLaMA o liczbie parametrów: 7B, 13B i 65B.

  <details>
    <summary>Full output</summary>

    ```python
    [chain/end] [chain:MapReduceDocumentsChain] [88.89s] Exiting Chain run with output:
    {
      "output_text": "<think>\n\n</think>\n\nDostępne są modele LLaMA o liczbie parametrów: 7B, 13B i 65B."
    }
    Question: Jakie modele LLaMa są dostępne? /no_think
    Response: {'input_documents': [Document(id='e6f2f97b-0624-4ae0-82be-267fddceca74', metadata={'Header 1': 'Rodzina modeli LLaMA – przegląd i wymagania sprzętowe (stan na sierpień2025 r.)', 'Header 2': 'LLaMA1 (2023)', 'source': 'llama.md'}, page_content='LLaMA1 była pierwszą publicznie dostępną rodziną modeli Mety.  Obejmuje warianty **7B**, **13B** i**65B** parametrów.  Modele te obsługują kontekst ok.2k tokenów ibyły udostępnione wyłącznie na zasadach badawczych.  \n* **Liczba parametrów isprzęt** – wariant 7B wprecyzjiFP16 wymaga ok.12–13GB pamięci VRAM, natomiast wersje zkwantyzacją (np.8‑bit lub 4‑bit) można uruchomić na kartach z6GB VRAMhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.  Model 13B wpełnej precyzji wymaga ~24GB VRAM, akwantyzowany – około 10GBhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.  Największy wariant 65B potrzebuje ponad 130GB VRAM izwykle wymaga wielu kart GPUhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.'), Document(id='891dcfb1-0d27-4208-8083-20f8da841348', metadata={'Header 1': 'Rodzina modeli LLaMA – przegląd i wymagania sprzętowe (stan na sierpień2025 r.)', 'Header 2': 'LLaMA1 (2023)', 'source': 'llama.md'}, page_content='LLaMA1 była pierwszą publicznie dostępną rodziną modeli Mety.  Obejmuje warianty **7B**, **13B** i**65B** parametrów.  Modele te obsługują kontekst ok.2k tokenów ibyły udostępnione wyłącznie na zasadach badawczych.  \n* **Liczba parametrów isprzęt** – wariant 7B wprecyzjiFP16 wymaga ok.12–13GB pamięci VRAM, natomiast wersje zkwantyzacją (np.8‑bit lub 4‑bit) można uruchomić na kartach z6GB VRAMhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.  Model 13B wpełnej precyzji wymaga ~24GB VRAM, akwantyzowany – około 10GBhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.  Największy wariant 65B potrzebuje ponad 130GB VRAM izwykle wymaga wielu kart GPUhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.'), Document(id='eab761b7-e20f-440c-b79f-8e11a8a9c16b', metadata={'Header 1': 'Rodzina modeli LLaMA – przegląd i wymagania sprzętowe (stan na sierpień2025 r.)', 'Header 2': 'LLaMA1 (2023)', 'source': 'llama.md'}, page_content='LLaMA1 była pierwszą publicznie dostępną rodziną modeli Mety.  Obejmuje warianty **7B**, **13B** i**65B** parametrów.  Modele te obsługują kontekst ok.2k tokenów ibyły udostępnione wyłącznie na zasadach badawczych.  \n* **Liczba parametrów isprzęt** – wariant 7B wprecyzjiFP16 wymaga ok.12–13GB pamięci VRAM, natomiast wersje zkwantyzacją (np.8‑bit lub 4‑bit) można uruchomić na kartach z6GB VRAMhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.  Model 13B wpełnej precyzji wymaga ~24GB VRAM, akwantyzowany – około 10GBhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.  Największy wariant 65B potrzebuje ponad 130GB VRAM izwykle wymaga wielu kart GPUhttps://www.bacloud.com/en/blog/163/guide-to-gpu-requirements-for-running-ai-models.html#:~:text=LLaMA%20,Meta%20AI.')], 'question': 'Jakie modele LLaMa są dostępne? /no_think', 'output_text': '<think>\n\n</think>\n\nDostępne są modele LLaMA o liczbie parametrów: 7B, 13B i 65B.'}
    ```
    
  </details>

- Q: Kto stworzył PLLuM? \
  A: Nie wiem.

  <details>
    <summary>Full output</summary>

    ```python
    [chain/end] [chain:MapReduceDocumentsChain] [46.74s] Exiting Chain run with output:
    {
      "output_text": "<think>\n\n</think>\n\nNie wiem."
    }
    Question: Kto stworzył PLLuM? /no_think
    Response: {'input_documents': [Document(id='e0f0eb98-26a5-4564-8496-aabd64a2c5cf', metadata={'Header 1': 'PLLuM– polski model językowy', 'Header 2': 'Zastosowania', 'source': 'pllum.md'}, page_content='PLLuM jest zaprojektowany jako podstawowa warstwa dla wielu zastosowań w języku polskim:  \n- **Generacja i analiza tekstu** – podstawowe zadania generacji, streszczania, tłumaczenia i analizy sentymentu; modele 8B i 12B zapewniają szybkie odpowiedzi, a 70B oferuje najwyższą jakość i kontekst do długich dokumentów.\n- **Asystenci dla administracji publicznej** – specjalne modele RAG potrafią odpowiadać na pytania dotyczące procedur administracyjnych czy ustaw, korzystając z dedykowanych baz dokumentówhttps://huggingface.co/CYFRAGOVPL/PLLuM-12B-instruct#:~:text=%2A%20Domain,information%20retrieval%20and%20question%20answering.\n- **Systemy pytanie‑odpowiedź (QA) i chat** – wersje „chat” i „instruct” są dostrojone do interakcji z użytkownikiem i generują bezpieczne, adekwatne odpowiedzi.'), Document(id='d6004b3c-f4c7-4e82-8801-d35f411a17f0', metadata={'Header 1': 'PLLuM– polski model językowy', 'Header 2': 'Zastosowania', 'source': 'pllum.md'}, page_content='PLLuM jest zaprojektowany jako podstawowa warstwa dla wielu zastosowań w języku polskim:  \n- **Generacja i analiza tekstu** – podstawowe zadania generacji, streszczania, tłumaczenia i analizy sentymentu; modele 8B i 12B zapewniają szybkie odpowiedzi, a 70B oferuje najwyższą jakość i kontekst do długich dokumentów.\n- **Asystenci dla administracji publicznej** – specjalne modele RAG potrafią odpowiadać na pytania dotyczące procedur administracyjnych czy ustaw, korzystając z dedykowanych baz dokumentówhttps://huggingface.co/CYFRAGOVPL/PLLuM-12B-instruct#:~:text=%2A%20Domain,information%20retrieval%20and%20question%20answering.\n- **Systemy pytanie‑odpowiedź (QA) i chat** – wersje „chat” i „instruct” są dostrojone do interakcji z użytkownikiem i generują bezpieczne, adekwatne odpowiedzi.'), Document(id='d2c8b8e7-3803-4ce7-a9d3-32c12876f0bc', metadata={'Header 1': 'PLLuM– polski model językowy', 'Header 2': 'Zastosowania', 'source': 'pllum.md'}, page_content='PLLuM jest zaprojektowany jako podstawowa warstwa dla wielu zastosowań w języku polskim:  \n- **Generacja i analiza tekstu** – podstawowe zadania generacji, streszczania, tłumaczenia i analizy sentymentu; modele 8B i 12B zapewniają szybkie odpowiedzi, a 70B oferuje najwyższą jakość i kontekst do długich dokumentów.\n- **Asystenci dla administracji publicznej** – specjalne modele RAG potrafią odpowiadać na pytania dotyczące procedur administracyjnych czy ustaw, korzystając z dedykowanych baz dokumentówhttps://huggingface.co/CYFRAGOVPL/PLLuM-12B-instruct#:~:text=%2A%20Domain,information%20retrieval%20and%20question%20answering.\n- **Systemy pytanie‑odpowiedź (QA) i chat** – wersje „chat” i „instruct” są dostrojone do interakcji z użytkownikiem i generują bezpieczne, adekwatne odpowiedzi.')], 'question': 'Kto stworzył PLLuM? /no_think', 'output_text': '<think>\n\n</think>\n\nNie wiem.'}
    ```
    
  </details>

- Q: Jaki model najlepiej działa na GPU z 24 GB VRAM? \
  A: Model Mini.

  <details>
    <summary>Full output</summary>

    ```python
    [chain/end] [chain:MapReduceDocumentsChain] [37.42s] Exiting Chain run with output:
    {
      "output_text": "<think>\n\n</think>\n\nModel Mini."
    }
    Question: Jaki model najlepiej działa na GPU z 24 GB VRAM? /no_think
    Response: {'input_documents': [Document(id='2e18e7c1-7bd7-444e-82af-d03059b58824', metadata={'Header 1': 'Modele Mistral– kompendium (stan na sierpień2025)', 'Header 2': 'Opisy poszczególnych modeli', 'Header 3': 'VoxtralSmall24B iVoxtralMini3B', 'source': 'mistal.md'}, page_content='Modele są gotowe do pobrania na HuggingFace; wymagają ok.60GBVRAM (Small) lub 8GB (Mini)https://docs.mistral.ai/getting-started/models/weights/#:~:text=Devstral,Mini%203B%203B%208.  Benchmarki publikowane wartykule pokazują, że Voxtral znacząco przewyższa Whisper large‑v3 iGPT‑4o mini wdokładności transkrypcji irozumieniahttps://mistral.ai/news/voxtral#:~:text=Voxtral%20comprehensively%20outperforms%20Whisper%20large,demonstrating%20its%20strong%20multilingual%20capabilities.'), Document(id='b2656e17-6cfe-4a42-a1cf-9140495423eb', metadata={'Header 1': 'Modele Mistral– kompendium (stan na sierpień2025)', 'Header 2': 'Opisy poszczególnych modeli', 'Header 3': 'VoxtralSmall24B iVoxtralMini3B', 'source': 'mistal.md'}, page_content='Modele są gotowe do pobrania na HuggingFace; wymagają ok.60GBVRAM (Small) lub 8GB (Mini)https://docs.mistral.ai/getting-started/models/weights/#:~:text=Devstral,Mini%203B%203B%208.  Benchmarki publikowane wartykule pokazują, że Voxtral znacząco przewyższa Whisper large‑v3 iGPT‑4o mini wdokładności transkrypcji irozumieniahttps://mistral.ai/news/voxtral#:~:text=Voxtral%20comprehensively%20outperforms%20Whisper%20large,demonstrating%20its%20strong%20multilingual%20capabilities.'), Document(id='cf5a7408-ecdb-4adc-83d1-d3400ff6422a', metadata={'Header 1': 'Modele Mistral– kompendium (stan na sierpień2025)', 'Header 2': 'Opisy poszczególnych modeli', 'Header 3': 'VoxtralSmall24B iVoxtralMini3B', 'source': 'mistal.md'}, page_content='Modele są gotowe do pobrania na HuggingFace; wymagają ok.60GBVRAM (Small) lub 8GB (Mini)https://docs.mistral.ai/getting-started/models/weights/#:~:text=Devstral,Mini%203B%203B%208.  Benchmarki publikowane wartykule pokazują, że Voxtral znacząco przewyższa Whisper large‑v3 iGPT‑4o mini wdokładności transkrypcji irozumieniahttps://mistral.ai/news/voxtral#:~:text=Voxtral%20comprehensively%20outperforms%20Whisper%20large,demonstrating%20its%20strong%20multilingual%20capabilities.')], 'question': 'Jaki model najlepiej działa na GPU z 24 GB VRAM? /no_think', 'output_text': '<think>\n\n</think>\n\nModel Mini.'}
    ```
    
  </details>

- Q: Kiedy powstał pierwszy model typu GPT? \
  A: W 2018 roku.

  <details>
    <summary>Full output</summary>

    ```python
    [chain/end] [chain:MapReduceDocumentsChain] [22.13s] Exiting Chain run with output:
    {
      "output_text": "<think>\n\n</think>\n\nW 2018 roku."
    }
    Question: Kiedy powstał pierwszy model typu GPT? /no_think
    Response: {'input_documents': [Document(id='82d4e2ce-bd3f-4f4f-9151-845416c361e8', metadata={'Header 1': 'Modele GPT – szczegółowy przegląd dla RAG (stan na sierpień 2025)', 'Header 2': '1. Ewolucja serii GPT', 'source': 'gpt.md'}, page_content='Przewyższa GPT‑3.5 wrozumieniu poleceń ikodowaniu. |'), Document(id='4b45bd68-bb04-49ae-bb84-157d130761b2', metadata={'Header 1': 'Modele GPT – szczegółowy przegląd dla RAG (stan na sierpień 2025)', 'Header 2': '1. Ewolucja serii GPT', 'source': 'gpt.md'}, page_content='Przewyższa GPT‑3.5 wrozumieniu poleceń ikodowaniu. |'), Document(id='248d722f-7cee-4095-ac91-98a8ed429c2f', metadata={'Header 1': 'Modele GPT – szczegółowy przegląd dla RAG (stan na sierpień 2025)', 'Header 2': '1. Ewolucja serii GPT', 'source': 'gpt.md'}, page_content='Przewyższa GPT‑3.5 wrozumieniu poleceń ikodowaniu. |')], 'question': 'Kiedy powstał pierwszy model typu GPT? /no_think', 'output_text': '<think>\n\n</think>\n\nW 2018 roku.'}
    ```
    
  </details>

- Q: Jakie są wymaganie sprzętowe dla modelu Llama 4 Scout? \
  A: Nie wiem.

  <details>
    <summary>Full output</summary>

    ```python
    [chain/end] [chain:MapReduceDocumentsChain] [38.64s] Exiting Chain run with output:
    {
      "output_text": "<think>\n\n</think>\n\nNie wiem."
    }
    Question: Jakie są wymaganie sprzętowe dla modelu Llama 4 Scout? /no_think
    Response: {'input_documents': [Document(id='ffdf65c9-fa79-4767-a4c2-31bc4ea3bd40', metadata={'Header 1': 'Rodzina modeli LLaMA – przegląd i wymagania sprzętowe (stan na sierpień2025 r.)', 'Header 2': 'LLaMA4 (kwiecień2025)', 'source': 'llama.md'}, page_content='* **Wyniki izastosowania** – Meta podaje, że Llama4Scout przewyższa Mistral3.1 oraz Gemma3 wzadaniach multimodalnych ioferuje najlepszy stosunek cena–jakość wswojej klasiehttps://ai.meta.com/blog/llama-4-multimodal-intelligence/#:~:text=,class%20performance%20to%20cost.  Llama4Maverick ma dorównywać GPT‑4o wbenchmarkach reasoningowych, ale wymaga dużej infrastruktury.  Modele te są przeznaczone do tworzenia asystentów multimodalnych (tekst+obraz), zaawansowanych aplikacji RAG zogromnym kontekstem (kilka milionów tokenów) isystemów budujących agentów zdługą pamięcią.'), Document(id='3ca300f7-92cc-4493-8091-914316404eff', metadata={'Header 1': 'Rodzina modeli LLaMA – przegląd i wymagania sprzętowe (stan na sierpień2025 r.)', 'Header 2': 'LLaMA4 (kwiecień2025)', 'source': 'llama.md'}, page_content='* **Wyniki izastosowania** – Meta podaje, że Llama4Scout przewyższa Mistral3.1 oraz Gemma3 wzadaniach multimodalnych ioferuje najlepszy stosunek cena–jakość wswojej klasiehttps://ai.meta.com/blog/llama-4-multimodal-intelligence/#:~:text=,class%20performance%20to%20cost.  Llama4Maverick ma dorównywać GPT‑4o wbenchmarkach reasoningowych, ale wymaga dużej infrastruktury.  Modele te są przeznaczone do tworzenia asystentów multimodalnych (tekst+obraz), zaawansowanych aplikacji RAG zogromnym kontekstem (kilka milionów tokenów) isystemów budujących agentów zdługą pamięcią.'), Document(id='264f63f7-42ce-42ac-87b0-e7a9c5ea38bc', metadata={'Header 1': 'Rodzina modeli LLaMA – przegląd i wymagania sprzętowe (stan na sierpień2025 r.)', 'Header 2': 'LLaMA4 (kwiecień2025)', 'source': 'llama.md'}, page_content='* **Wyniki izastosowania** – Meta podaje, że Llama4Scout przewyższa Mistral3.1 oraz Gemma3 wzadaniach multimodalnych ioferuje najlepszy stosunek cena–jakość wswojej klasiehttps://ai.meta.com/blog/llama-4-multimodal-intelligence/#:~:text=,class%20performance%20to%20cost.  Llama4Maverick ma dorównywać GPT‑4o wbenchmarkach reasoningowych, ale wymaga dużej infrastruktury.  Modele te są przeznaczone do tworzenia asystentów multimodalnych (tekst+obraz), zaawansowanych aplikacji RAG zogromnym kontekstem (kilka milionów tokenów) isystemów budujących agentów zdługą pamięcią.')], 'question': 'Jakie są wymaganie sprzętowe dla modelu Llama 4 Scout? /no_think', 'output_text': '<think>\n\n</think>\n\nNie wiem.'}
    ```
    
  </details>

## Summary

Overall, the RAG pipeline successfully retrieves relevant information from the knowledge base and generates answers to the provided questions. The system demonstrates the capabilities of LangChain and Hugging Face Transformers in building a question-answering system that can leverage external knowledge sources. Honestly, answers are not perfect, but they are relevant to the questions asked. The system can be further improved by using larger models, increasing the retrieval depth, and implementing additional validation steps.

## Possible improvements

Considering larger hardware possibilities, there are several improvements that could be made to the RAG pipeline:
- increasing models sizes (thus parameters) to improve quality of the answers,
- increasing value of `k` parameter in retrieval step to retrieve more information from the knowledge base,
- implement reranking [step](https://huggingface.co/Qwen/Qwen3-Reranker-8B) to better validate context before passing it to the chat model,
- implement [external graph knowledge base](https://python.langchain.com/docs/integrations/providers/neo4j/) to enhance the retrieval step based on knowledge/fact graph entities.

## References
- [Langchain](https://www.langchain.com)
- [Hugging Face](https://huggingface.co)
- [Knowledge Graphs](https://arxiv.org/abs/2003.02320)
- [Fact knowledge graphs](https://arxiv.org/abs/2305.06590)