import os
import time
from app.file_utils import load_hidden_documents
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Global variables
VECTOR_STORE = None
FILE_MOD_TIMES = {}
LAST_RELOAD_TIME = 0  # To track the last reload time
RELOAD_INTERVAL = 120  # Reload interval in seconds (e.g., 2 minutes)

# Create vector store from document texts
def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

# Get modification time for a specific file
def get_file_mod_time(file_path):
    """Get the modification time of a specific file."""
    return os.path.getmtime(file_path)

# Reload vector store if any file has been modified
def reload_vector_store_if_needed(directory="hidden_docs"):
    global VECTOR_STORE, FILE_MOD_TIMES, LAST_RELOAD_TIME

    # Only proceed if the reload interval has passed
    current_time = time.time()
    if current_time - LAST_RELOAD_TIME < RELOAD_INTERVAL:
        # Not enough time has passed for reloading
        return VECTOR_STORE

    # Get the current modification times of files in the directory
    current_mod_times = {}

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            current_mod_times[filename] = get_file_mod_time(file_path)

    # Check for changes in modification times
    files_to_reload = []

    for filename, mod_time in current_mod_times.items():
        if filename not in FILE_MOD_TIMES or FILE_MOD_TIMES[filename] != mod_time:
            files_to_reload.append(filename)

    # If there are modified files, reload the corresponding documents
    if files_to_reload:
        print(f"Reloading files: {files_to_reload}")
        document_texts = load_hidden_documents(directory)
        VECTOR_STORE = create_vector_store(document_texts)

        # Update the FILE_MOD_TIMES dictionary with new modification times
        FILE_MOD_TIMES = current_mod_times

        # Update the last reload time
        LAST_RELOAD_TIME = current_time

    return VECTOR_STORE
