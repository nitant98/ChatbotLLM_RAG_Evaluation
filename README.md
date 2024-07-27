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
- [Contributing](#contributing)
- [License](#license)

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
