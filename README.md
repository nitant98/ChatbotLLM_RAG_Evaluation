# LOTR Conversational RAG Chatbot

This project is a Conversational Retrieval-Augmented Generation (RAG) Chatbot designed to answer questions about J.R.R. Tolkien's "The Lord of the Rings". The chatbot utilizes a combination of a pre-trained language model and a vector store for document retrieval to provide accurate and contextually relevant answers.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Challenges](#challenges)

## Overview

The chatbot leverages the following components:
- **LangChain**: To create and manage the retrieval and generation chains.
- **Pinecone**: For storing and retrieving document embeddings.
- **HuggingFace Transformers**: For the language model and embeddings.
- **Streamlit**: To create the web interface for the chatbot.

## Project Structure

├── LOTR_books/
│ └── Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010.pdf
├── app.py
├── config.py
├── create_embeddings.py
├── evaluation_metrics.py
├── evaluation_metrics_fine_tune.py
├── model_setup.py
├── qa_setup.py
├── qa_setup_old.py
├── model_setup_old.py
├── requirements.txt
└── README.md

### Files
- **`LOTR_books/`**: Directory containing the LOTR PDF book.
- **`app.py`**: Streamlit application file to run the chatbot interface.
- **`config.py`**: Configuration file for setting parameters.
- **`create_embeddings.py`**: Script to create and store document embeddings in Pinecone.
- **`evaluation_metrics.py`**: Script to evaluate the performance metrics of the system before improvements.
- **`evaluation_metrics_fine_tune.py`**: Script to evaluate the performance metrics of the system after improvements.
- **`model_setup.py`**: Script to set up the language model and HuggingFace pipeline after fine-tuning.
- **`qa_setup.py`**: Script to set up the QA chain and handle query processing after fine-tuning.
- **`qa_setup_old.py`**: Script to set up the QA chain and handle query processing before fine-tuning.
- **`model_setup_old.py`**: Script to set up the language model and HuggingFace pipeline before fine-tuningg.
- **`requirements.txt`**: Contains the list of dependencies.
- **`README.md`**: This file.

## Setup Instructions

### Prerequisites
Ensure you have Python 3.8 or higher installed. You'll also need a Pinecone account and API key.

### Install Dependencies

```sh
pip install -r requirements.txt
```
