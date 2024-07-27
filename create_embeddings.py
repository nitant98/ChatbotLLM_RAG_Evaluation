# create_embeddings.py
import glob
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from config import CFG
import os
import pinecone
import torch
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

def create_and_store_embeddings():
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
            dimension=768,  # Adjust the dimension according to your embedding size
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_host  # Adjust as per your environment settings
            )
        )

    # Connect to the index
    index = pc.Index(index_name)


 # Verify the directory path and list files
    if not os.path.exists(CFG.PDFs_path):
        print(f"Directory {CFG.PDFs_path} does not exist.")
        return

    pdf_files = glob.glob(os.path.join(CFG.PDFs_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in directory {CFG.PDFs_path}.")
        return

    print(f"Found {len(pdf_files)} PDF files in directory {CFG.PDFs_path}.")

    # Set up the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.split_chunk_size,
        chunk_overlap=CFG.split_overlap
    )

    # Load and split documents
    documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            doc = loader.load()
            documents.extend(doc)
            print(f"Successfully loaded {pdf_file}")
        except Exception as e:
            print(f"Error loading file {pdf_file}: {e}")

    print(f"Loaded {len(documents)} documents")
    if len(documents) > 0:
        print(f"Sample document content: {documents[0].page_content[:500]}")  # Print first 500 characters of the first document

    texts = text_splitter.split_documents(documents)

    print(f"Split documents into {len(texts)} chunks")
    if len(texts) > 0:
        print(f"Sample chunk content: {texts[0].page_content[:500]}")  # Print first 500 characters of the first chunk

    # Ensure texts are correctly formatted as pairs for InstructorEmbedding
    instruction_pairs = [("Represent the meaning of this text", text.page_content) for text in texts]
    print(f"Formatted {len(instruction_pairs)} instruction-text pairs")

    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SentenceTransformer directly
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(CFG.embeddings_model_repo, device=device)

    # Function to embed texts
    def embed_texts(embedding_model, instruction_pairs):
        embeddings = []
        batch_size = 32  # Adjust based on your memory capacity
        for i in range(0, len(instruction_pairs), batch_size):
            batch = [pair[1] for pair in instruction_pairs[i:i+batch_size]]
            batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True, device=device)
            embeddings.extend(batch_embeddings)
        return embeddings

    # Embed documents
    embedded_texts = embed_texts(embedding_model, instruction_pairs)

    # Function to batch upsert vectors to Pinecone
    def upsert_vectors_in_batches(index, vectors, batch_size=100):
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
    '''
    # Prepare vectors for upsert
    vectors = [(str(i), embedded_texts[i].cpu().numpy().tolist()) for i in range(len(embedded_texts))]

    # Upsert vectors to Pinecone in batches
    upsert_vectors_in_batches(index, vectors, batch_size=100)

    print("Embeddings created and stored in Pinecone")
    '''

    # Prepare vectors with metadata for upsert
    vectors = [(str(i), embedded_texts[i].cpu().numpy().tolist(), {"text": texts[i].page_content}) for i in range(len(embedded_texts))]

    # Upsert vectors to Pinecone in batches
    upsert_vectors_in_batches(index, vectors, batch_size=100)



if __name__ == "__main__":

    create_and_store_embeddings()
