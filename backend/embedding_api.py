import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

def get_embedding_model():
    """
    Returns the LangChain wrapper for Hugging Face Inference API embeddings.
    Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "your_huggingface_api_token_here":
        raise ValueError("Please set a valid HF_TOKEN in your .env file.")
        
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token,
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return embeddings
