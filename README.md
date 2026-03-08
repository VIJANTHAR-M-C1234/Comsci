# AI-01 – Hinglish / Tanglish NCERT Doubt-Clearing Chatbot.

An educational AI system that helps school students (Class 6 - 12) solve their NCERT textbook doubts. The chatbot uses Retrieval-Augmented Generation (RAG) and open-source models remotely on Hugging Face using the Inference API. It supports mixed languages like Hinglish and Tanglish.

## Setup Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   Create a `.env` file in the root directory and add your Hugging Face API Token:
   ```env
   HF_TOKEN=your_token_here
   ```

3. **Ingest NCERT PDFs**
   Place your NCERT PDF files into the `data/ncert_pdfs/` directory. Then run the ingestion script:
   ```bash
   python backend/ingest.py
   ```
   This will extract text, chunk it, request embeddings from Hugging Face, and store the vectors in the `vector_db/` directory.

4. **Run the Chatbot Application**
   Start the Streamlit UI:
   ```bash
   streamlit run frontend/app.py
   ```

## Architecture
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Vector DB**: Chroma
- **Frontend**: Streamlit
