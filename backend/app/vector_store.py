import os
from app.file_utils import load_hidden_documents
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Global variable to store vector store
VECTOR_STORE = None
FILE_MOD_TIMES = {}

# Create vector store from document texts
def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

# Get modification times for documents
def get_file_mod_times(directory):
    """Get the modification times of all files in the directory."""
    return {
        f: os.path.getmtime(os.path.join(directory, f))
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))  # Ensure it's a file, not a directory
    }

# Reload vector store if needed
def reload_vector_store_if_needed(directory="hidden_docs"):
    global VECTOR_STORE, FILE_MOD_TIMES

    current_mod_times = get_file_mod_times(directory)

    # Check if files have been modified
    if FILE_MOD_TIMES != current_mod_times:
        FILE_MOD_TIMES = current_mod_times
        document_texts = load_hidden_documents(directory)
        VECTOR_STORE = create_vector_store(document_texts)

    return VECTOR_STORE
