Llama Fine-Tuning and RAG Approaches for Tourism in Morocco
This repository contains notebooks and datasets for fine-tuning Llama models and applying Retrieval Augmented Generation (RAG) approaches for a chatbot focused on tourism in Morocco. The training dataset (tourmaroc.csv) contains 17,000 rows, while the test dataset (test.csv) consists of 68 rows.

Repository Contents
llama2-finetuning.ipynb: Notebook for fine-tuning Llama 2 model using the tourmaroc dataset.
llama3-finetuning.ipynb: Notebook for fine-tuning Llama 3 model using the tourmaroc dataset.
rag with faiss.ipynb: Notebook for implementing Retrieval Augmented Generation (RAG) with FAISS for document retrieval.
rag with qdrant.ipynb: Notebook for implementing Retrieval Augmented Generation (RAG) with Qdrant for document retrieval.
test.csv: The test dataset used for evaluating the model, consisting of 68 rows.
tourmaroc.csv: The main training dataset, containing 17,000 rows of data related to tourism in Morocco.

The fine-tuned models are hosted on Hugging Face and can be accessed via the following links:
LLaMA 2 tourism Model: https://huggingface.co/laila1234/tourllama2
LLaMA 3 tourism Model: https://huggingface.co/laila1234/tourllama3.1
LLaMA 3.1 tourism Model: coming soon
