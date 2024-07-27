'''                                 # WORKINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
# qa_setup.py
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from model_setup import llm
from config import CFG, index

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone vector store
vector_db = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Create prompt template
system_prompt = """
Use the given context to answer the question.
If you don't know the answer, say you don't know.
Use three sentences maximum and keep the answer concise.
Context: {context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents and answer questions
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 3}), question_answer_chain)

def llm_ans(query):
    try:
        # Fetch relevant documents
        print("Fetching relevant documents...")
        docs = vector_db.similarity_search(query=query)
        print(f"Retrieved {len(docs)} documents.")
        
        # Create context from documents
        context = "\n".join([doc.page_content for doc in docs])
        print(f"Context created with length {len(context)} characters.")
        
        # Structure the input correctly
        structured_input = {"input": query, "context": context}
        print(f"Structured Input: {structured_input}")
        
        # Invoke the chain with the structured input
        response = chain.invoke(structured_input)
        
        # Print the response type and content
        print(f"LLM Response Type: {type(response)}")
        print(f"LLM Response Content: {response}")
        
        print("LLM response generated.")
        return response
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

# Test the function
question = "Who is the King of Rohan?"
print(llm_ans(question))
'''
# qa_setup.py
# qa_setup.py
# qa_setup.py

from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CFG, index
from model_setup import get_llm_response

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone vector store
vector_db = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Create prompt template
system_prompt = """
Use the given context to answer the question.
If you don't know the answer, say you don't know.
Use three sentences maximum and keep the answer concise.
Context: {context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents and answer questions
question_answer_chain = create_stuff_documents_chain(get_llm_response, prompt)
chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 3}), question_answer_chain)

def llm_ans(query):
    try:
        # Fetch relevant documents
        print("Fetching relevant documents...")
        docs = vector_db.similarity_search(query=query)
        print(f"Retrieved {len(docs)} documents.")
        
        # Create context from documents
        context = "\n".join([doc.page_content for doc in docs])
        print(f"Context created with length {len(context)} characters.")
        
        # Structure the input correctly
        structured_input = {
            "input": query,
            "context": context
        }
        print(f"Structured Input: {structured_input}")
        
        # Format the prompt
        formatted_prompt = f"Context: {context}\n\nQuestion: {query}"
        
        # Get the LLM response
        response = get_llm_response(formatted_prompt)
        
        print("LLM response generated.")
        
        # Return a structured result
        result = {
            "query": query,
            "context": context,
            "retrieved_documents": docs,
            "generated_answer": response
        }
        return result
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}
