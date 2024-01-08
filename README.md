# End-to-End LLM-based Scalable RAG-powered QA App

This repo implements an ***Retrival Augmented Generation (RAG)-powered LLM-based Open Generative (or Extractive) context-aware Question-Answering (QA)*** App that:

- Takes as input a new `query` (or `question`);
- Implements ***vector similarity search*** within the ***embedding space*** by seeking ***relevant contexts*** corresponding to the incoming `query` in the ***vector database***;
- Passes the ***relevant contexts*** as well as the `input query` to LLM;
- LLM then produces the `answer` to the input `query` while being aware of the ***relevant contexts*** related to the requested `query`.

The below diagram shows the _architectural design_ of this ***RAG-powered LLM App***:

![App Architecture](https://github.com/fork123aniket/LLM-RAG-powered-QA-App/assets/92912434/0387ac34-c876-4987-9400-9c0b9acc2934)

## Requirements
- `Python`
- `Streamlit`
- `PEFT` (for Parameter-Efficient Fine-Tuning)
- `Accelerate`
- `Ray` (for distributed LLM Fine-Tuning)
- `Datasets`
- `Transformers`
- `PyTorch`
- `Numpy`
- `Scikit-Learn`
- `Deta` (To access Deta Vector Database)
- `Langchain`
- `FastAPI` (To serve production-ready LLM App)

## Data
[***Squad***](https://huggingface.co/datasets/squad/viewer/plain_text/train?row=0) dataset is used to fine-tune [***Eleuther AI's GPT-Neo 20B***](https://huggingface.co/EleutherAI/gpt-neox-20b) LLM model, which comprises `Title`, `Question`, `Answer`, and `Context` for each of the `98.2k` dataset `IDs`.

## LLM Training and Serving
- The `Fine-Tuning` process for `GPT-Neo` LLM model can be found in `finetune.py` file.
- The code to create ***RAG-powered LLM Agent*** for `QA` task can be seen in `qa_agent.py` file.
- To build the ***agent*** as ***production-ready API*** for `QA` task, it's worth delving deep into `serve.py` file.
- To seek prospects of using `Streamlit` to deploy the LLM app, head to `streamlit.py` file.
- All hyperparameters to control `fine-tuning` of the model are provided in the given `config.py` file.

## App Usage
To learn more about how to use this ***LLM RAG-powered QA App***, consider watching the following video:

[RAG-powered LLM App.webm](https://github.com/fork123aniket/LLM-RAG-powered-QA-App/assets/92912434/c003342c-c337-44d1-a29d-b6c554eaabf9)
