import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path for execution as script 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf

# Increase pypdf decompression limit to handle large textbook PDFs
pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = 500000000

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ncert_pdfs")

def load_and_chunk_pdfs():
    """
    Loads all PDFs from the ncert_pdfs directory and splits them into smaller sections.
    """
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} does not exist. Creating it...")
        os.makedirs(DATA_DIR, exist_ok=True)
        return []

    print(f"Loading PDFs from {DATA_DIR}...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found. Please add PDF files to data/ncert_pdfs/")
        return []

    print(f"Loaded {len(documents)} pages. Splitting into chunks...")
    
    # Chunking strategy for context retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

if __name__ == "__main__":
    from backend.retriever import store_chunks, get_connection_info

    conn = get_connection_info()
    print(f"\n[ingest] Target database: {conn['mode']} - {conn['detail']}\n")

    chunks = load_and_chunk_pdfs()
    if chunks:
        store_chunks(chunks)
        print(f"\n[OK] Vector database ingestion completed successfully into {conn['mode']}.")
