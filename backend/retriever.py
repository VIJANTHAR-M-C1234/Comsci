import os
import chromadb
from langchain_community.vectorstores import Chroma
from backend.embedding_api import get_embedding_model
from dotenv import load_dotenv

load_dotenv()

# ─── Local fallback path ───────────────────────────────────────────────────────
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_db")

# ─── Chroma Cloud credentials (from .env) ──────────────────────────────────────
CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT   = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")
COLLECTION_NAME = "ncert_docs"


def _is_cloud_configured() -> bool:
    """Return True if all Chroma Cloud credentials are present in .env."""
    return bool(CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE)


def get_chroma_client():
    """
    Returns a Chroma client.
    - Uses Chroma Cloud if CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE are set.
    - Falls back to local PersistentClient otherwise.
    """
    if _is_cloud_configured():
        print("[retriever] 🌐 Connecting to Chroma Cloud...")
        client = chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY,
        )
        print("[retriever] ✅ Connected to Chroma Cloud.")
        return client, "cloud"
    else:
        print("[retriever] 💾 Using local Chroma (persistent).")
        client = chromadb.PersistentClient(path=DB_DIR)
        return client, "local"


def get_vector_store():
    """
    Initializes and returns the LangChain Chroma vector store,
    connected to either Chroma Cloud or local persistent storage.
    """
    embeddings = get_embedding_model()
    chroma_client, mode = get_chroma_client()

    vector_store = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vector_store, mode


def store_chunks(chunks):
    """
    Stores document chunks into the vector database
    (Cloud or local, whichever is configured).
    """
    print("[retriever] Initializing vector store for ingestion...")
    vector_store, mode = get_vector_store()

    print(f"[retriever] Storing {len(chunks)} chunks into Chroma ({mode})...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}...")
        vector_store.add_documents(batch)

    print(f"[retriever] ✅ Successfully stored chunks into Chroma ({mode}).")


def get_retriever():
    """
    Returns the retriever interface for searching the vector database.
    Raises FileNotFoundError if local DB is missing and cloud is not configured.
    """
    if not _is_cloud_configured():
        if not os.path.exists(DB_DIR):
            raise FileNotFoundError(
                "Local Vector DB not found AND Chroma Cloud is not configured.\n"
                "Either run backend/ingest.py to build the local DB,\n"
                "or set CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE in .env for cloud."
            )

    vector_store, mode = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": 3}), mode


def get_connection_info() -> dict:
    """
    Returns current Chroma connection details for UI display.
    """
    if _is_cloud_configured():
        return {
            "mode": "☁️ Chroma Cloud",
            "detail": f"Tenant: {CHROMA_TENANT} | DB: {CHROMA_DATABASE}",
            "is_cloud": True,
        }
    else:
        return {
            "mode": "💾 Local (Persistent)",
            "detail": f"Path: {DB_DIR}",
            "is_cloud": False,
        }
