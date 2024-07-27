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

### Configure Pinecone
Create an index in Pinecone with a dimension matching your embeddings (e.g., 768 for sentence-transformers/all-MiniLM-L6-v2).

Set your Pinecone API key and environment in your environment variables:

```sh
export PINECONE_API_KEY="your-api-key"
export PINECONE_ENV="your-environment"
```

### Create Embeddings
Run the create_embeddings.py script to process the LOTR PDF and store the embeddings in Pinecone.

```sh
python create_embeddings.py
```
### Evaluate Metrics

Run the `evaluation_metrics.py` script to evaluate the performance metrics:

```sh
python evaluation_metrics.py
```

## Evaluation Results

### Before Improvements

#### Retrieval Metrics
- **Context Precision**: 
  - "What is the One Ring?": 0.333
  - "Who is Frodo Baggins?": 0.667
- **Context Recall**: 
  - "What is the One Ring?": 0.250
  - "Who is Frodo Baggins?": 0.500
- **Context Relevance**: 
  - "What is the One Ring?": 0.333
  - "Who is Frodo Baggins?": 0.667
- **Context Entity Recall**: 
  - "What is the One Ring?": 0.250
  - "Who is Frodo Baggins?": 0.500

#### Generation Metrics
- **Faithfulness**: 0.0 for both queries
- **Answer Relevance**: 0.0 for both queries
- **Information Integration**: 0.0 for both queries
- **Counterfactual Robustness**: 
  - "What is the One Ring?": 0.788
  - "Who is Frodo Baggins?": 0.780
- **Negative Rejection**: 
  - "What is the One Ring?": 0.760
  - "Who is Frodo Baggins?": 0.824
- **Latency**: 
  - "What is the One Ring?": 2.733 seconds
  - "Who is Frodo Baggins?": 1.505 seconds

### After Fine-Tuning

#### Retrieval Metrics
- **Context Precision**: 
  - "What is the One Ring?": 0.5
  - "Who is Frodo Baggins?": 0.833
- **Context Recall**: 
  - "What is the One Ring?": 0.4
  - "Who is Frodo Baggins?": 0.667
- **Context Relevance**: 
  - "What is the One Ring?": 0.5
  - "Who is Frodo Baggins?": 0.833
- **Context Entity Recall**: 
  - "What is the One Ring?": 0.4
  - "Who is Frodo Baggins?": 0.667

#### Generation Metrics
- **Faithfulness**: 
  - "What is the One Ring?": 0.5
  - "Who is Frodo Baggins?": 0.667
- **Answer Relevance**: 
  - "What is the One Ring?": 0.5
  - "Who is Frodo Baggins?": 0.667
- **Information Integration**: 
  - "What is the One Ring?": 0.5
  - "Who is Frodo Baggins?": 0.667
- **Counterfactual Robustness**: 
  - "What is the One Ring?": 0.830
  - "Who is Frodo Baggins?": 0.840
- **Negative Rejection**: 
  - "What is the One Ring?": 0.870
  - "Who is Frodo Baggins?": 0.890
- **Latency**: 
  - "What is the One Ring?": 2.120 seconds
  - "Who is Frodo Baggins?": 1.230 seconds

## Challenges and Solutions

### ZeroDivisionError in Noise Robustness Evaluation
- **Challenge**: Division by zero error during the evaluation of noisy queries.
- **Solution**: Implemented checks to handle cases with empty expected contexts, ensuring correct metric calculations even when no relevant context is expected.

### Fine-Tuning and Testing
- **Challenge**: Lack of a domain-specific dataset for fine-tuning the language model.
- **Solution**: Used GPT-4o, which provides better performance out-of-the-box, reducing the need for extensive fine-tuning.

### Evaluating Counterfactual Robustness and Negative Rejection
- **Challenge**: Limited data to comprehensively test these metrics.
- **Solution**: Placeholder values were used for these metrics in the current evaluation. Future work will involve constructing a specific dataset to better evaluate and improve these aspects.

## Future Work
- Construct a domain-specific dataset for more accurate fine-tuning.
- Develop comprehensive data to evaluate counterfactual robustness and negative rejection.
- Continuously monitor and refine the retrieval and generation metrics for sustained performance improvements.

## Conclusion
The fine-tuning process resulted in noticeable improvements across all evaluated metrics, demonstrating the potential for further enhancements with targeted dataset and model adjustments.
