import os
import time
import pickle
from hashlib import md5
from app.file_utils import load_hidden_documents
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Global variables
VECTOR_STORE = None
FILE_HASHES = {}  # Store file hashes and modification times to track changes
LAST_RELOAD_TIME = 0  # To track the last reload time
RELOAD_INTERVAL = 120  # Reload interval in seconds (e.g., 2 minutes)
CACHE_PATH = "vector_store_cache.pkl"  # Path for caching the vector store

# Helper function to get file hash (MD5)
def get_file_hash(file_path):
    """Generate MD5 hash of the file content."""
    hash_md5 = md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_mod_time(file_path):
    """Get the modification time of a specific file."""
    return os.path.getmtime(file_path)

# Create vector store from document texts
def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

# Check if the cache is valid
def is_cache_valid():
    """Check if the cache file exists and is up to date."""
    if not os.path.exists(CACHE_PATH):
        return False
    # Check the last modified time of the cache file
    cache_mod_time = os.path.getmtime(CACHE_PATH)
    current_time = time.time()
    if current_time - cache_mod_time < RELOAD_INTERVAL:
        return True
    return False

# Load vector store from the cache
def load_vector_store_from_cache():
    """Load the vector store from the cache file."""
    with open(CACHE_PATH, 'rb') as cache_file:
        return pickle.load(cache_file)

# Save the vector store to the cache
def save_vector_store_to_cache(vector_store):
    """Save the vector store to the cache file."""
    with open(CACHE_PATH, 'wb') as cache_file:
        pickle.dump(vector_store, cache_file)

# Reload vector store if needed
def reload_vector_store_if_needed(directory="hidden_docs"):
    global VECTOR_STORE, FILE_HASHES, LAST_RELOAD_TIME

    # Only proceed if the reload interval has passed
    current_time = time.time()
    if current_time - LAST_RELOAD_TIME < RELOAD_INTERVAL:
        # Not enough time has passed for reloading
        return VECTOR_STORE

    # Check if the cache is valid
    if is_cache_valid():
        # Load the vector store from the cache
        VECTOR_STORE = load_vector_store_from_cache()
        print("Loaded vector store from cache.")
    else:
        # Get the current hashes and modification times of files in the directory
        current_file_data = {}

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):  # Ensure it's a file
                current_file_data[filename] = {
                    "hash": get_file_hash(file_path),
                    "mod_time": get_file_mod_time(file_path)
                }

        # Check for changes in the file hashes or modification times
        files_to_reload = []

        for filename, file_data in current_file_data.items():
            if filename not in FILE_HASHES or FILE_HASHES[filename] != file_data:
                files_to_reload.append(filename)

        # If there are modified or new files, reload the corresponding documents
        if files_to_reload:
            print(f"Reloading files: {files_to_reload}")
            try:
                document_texts = load_hidden_documents(directory)
                VECTOR_STORE = create_vector_store(document_texts)

                # Update the FILE_HASHES dictionary with the new file data
                FILE_HASHES = current_file_data

                # Save the updated vector store to the cache
                save_vector_store_to_cache(VECTOR_STORE)

                # Update the last reload time
                LAST_RELOAD_TIME = current_time
            except Exception as e:
                print(f"Error while reloading vector store: {e}")
                return VECTOR_STORE  # Return the existing vector store in case of error
        else:
            print("No file changes detected. Using cached vector store.")
    
    return VECTOR_STORE
