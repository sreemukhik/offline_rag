import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Configuration
DATA_DIR = "data"
CHUNK_SIZE = 2000 # Approx 500 tokens
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.pkl"

def load_documents():
    """Lengths load PDFs from the data directory."""
    pdf_files = glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True)
    documents = []
    print(f"Found {len(pdf_files)} PDF files in {DATA_DIR}...")
    
    for file_path in pdf_files:
        try:
            print(f"Loading {file_path}...")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return documents

def split_text(documents):
    """Splits documents into chunks."""
    if not documents:
        print("No documents successfully loaded.")
        return []
        
    print(f"Splitting {len(documents)} document pages...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} text chunks.")
    return chunks

def generate_embeddings_and_index(chunks):
    """Generates embeddings and stores them in FAISS."""
    if not chunks:
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # Initialize SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Extract text content from chunks
    texts = [chunk.page_content for chunk in chunks]
    
    print("Generating embeddings (this may take a while)...")
    embeddings = model.encode(texts)
    
    # Convert to numpy array for FAISS
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings_np.shape[1]
    print(f"Creating FAISS index (Dimension: {dimension}, IndexFlatL2)...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    # Save index
    print(f"Saving index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    
    # Save metadata (text and source info) to retrieve later
    # FAISS only stores vectors, we need a way to map ID back to text
    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(chunks, f)
        
    print("Ingestion complete.")

def main():
    # 1. Load PDFs
    documents = load_documents()
    
    # 2. Split into chunks
    chunks = split_text(documents)
    
    if chunks:
        # 3. Generate Embeddings & Save Index
        generate_embeddings_and_index(chunks)
    else:
        print("No content to process. Please add PDF files to the 'data' directory.")

if __name__ == "__main__":
    main()
