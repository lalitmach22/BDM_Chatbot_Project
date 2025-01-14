import os
from .extract_texts import logger
from threading import Lock
from .documents import store_file_hashes_in_supabase
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from .extract_texts import load_hidden_documents
from .embeddings import store_embeddings_in_supabase, load_embeddings_from_supabase


VECTOR_STORE_PATH = "faiss_index"
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# In-memory storage for vector store and file modification times
in_memory_store = {
    "vector_store": None,
    "file_mod_times": None
}

# Thread lock for thread safety
store_lock = Lock()

def create_vector_store(document_texts):
    """Create a FAISS vector store from the provided document texts."""
    vector_store = FAISS.from_texts(document_texts, embedder)
    logger.info("Vector store created successfully.")
    return vector_store

def load_or_build_vector_store(directory, supabase_client):
    """Load the existing vector store if available, otherwise build a new one."""
    # Check for new or modified files
    new_or_modified_files = store_file_hashes_in_supabase(directory, supabase_client)
    logger.info(f"Found {len(new_or_modified_files)} new or modified files.")

    # Check if there's an existing vector store
    if os.path.exists(VECTOR_STORE_PATH):
        logger.info("Loading existing vector store...")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedder, allow_dangerous_deserialization=True)
    else:
        vector_store = None

    # Case 1: New or modified files exist and vector store exists
    if new_or_modified_files and vector_store:
        logger.info("Building vector store for new files and merging with existing store...")
        document_texts = load_hidden_documents(directory, files=new_or_modified_files)
        if not document_texts:
            logger.warning("No content found in new or modified files.")
            return vector_store  # Return the existing vector store
        new_vector_store = FAISS.from_texts(document_texts, embedder)
        vector_store.merge_from(new_vector_store)

    # Case 2: No new files, but an existing vector store exists
    elif not new_or_modified_files and vector_store:
        logger.info("No new files. Using existing vector store.")
        return vector_store

    # Case 3: New files exist and no vector store, OR no new files and no vector store
    elif (new_or_modified_files and not vector_store) or (not new_or_modified_files and not vector_store):
        logger.info("Building new vector store from all documents.")
        document_texts = load_hidden_documents(directory)
        if not document_texts:
            logger.warning("No documents found in the directory to build a vector store.")
            return None
        vector_store = create_vector_store(document_texts)

    # Save the updated or newly created vector store
    vector_store.save_local(VECTOR_STORE_PATH)
    logger.info("Vector store updated and saved.")
    return vector_store

def reload_vector_store_if_needed(directory, supabase_client):
    """Reload the vector store if any files in the directory have changed."""
    with store_lock:  # Ensure thread safety
        vector_store = load_or_build_vector_store(directory, supabase_client)
        if not vector_store:
            if in_memory_store.get("vector_store") is None:
                logger.error("Failed to load or build vector store. No vector store available.")
                return None
            logger.info("Using existing in-memory vector store.")
        else:
            in_memory_store["vector_store"] = vector_store
            logger.info("Vector store reloaded and stored in memory.")
    
    return in_memory_store.get("vector_store", None)