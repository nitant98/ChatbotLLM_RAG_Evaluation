'''

import os
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Configuration class
class CFG:
    embeddings_model_repo = "sentence-transformers/all-MiniLM-L6-v2"
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.0
    #embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'
    k = 6
    PDFs_path = 'LOTR_books/'

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = os.getenv("PINECONE_HOST")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Define the index name
index_name = "bookvectors"

# Check if the index exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric='euclidean',
        spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_host  
            )
        )

    # Connect to the index
index = pc.Index(index_name)

def initialize_pinecone(api_key, host_name):
        pc = Pinecone(api_key=api_key)
        index = pc.Index(host = host_name)


initialize_pinecone(pinecone_api,  pinecone_host )
'''

import os
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# config.py

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = os.getenv("PINECONE_HOST")
openapi_key = os.getenv("openai_api")

class CFG:
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.0
    openai_api_key = openapi_key



# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Define the index name
index_name = "bookvectors"

# Ensure the index variable is properly set
index = pc.Index(index_name)



