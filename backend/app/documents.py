import os
import hashlib
from .extract_texts import logger

def generate_file_hash(filepath):
    """Generate a SHA-256 hash for the content of a file."""
    with open(filepath, "rb") as f:
        file_content = f.read()
    return hashlib.sha256(file_content).hexdigest()

def store_file_hashes_in_supabase(directory, supabase_client):
    """Store filenames and their hashes in Supabase."""
    existing_hashes = load_file_hashes_from_supabase(supabase_client)
    new_or_modified_files = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue  # Skip directories

        file_hash = generate_file_hash(filepath)

        # Check if file hash already exists in Supabase
        if filename not in existing_hashes or existing_hashes[filename] != file_hash:
            new_or_modified_files.append((filename, file_hash))
    
    # Store new or modified file hashes in Supabase
    for filename, file_hash in new_or_modified_files:
        data = {"filename": filename, "hash": file_hash}
        response = supabase_client.table("file_hashes").upsert(data).execute()
        if response:
            logger.info(f"Stored hash for file: {filename}")
        else:
            logger.error(f"Failed to store hash for file: {filename} | Response: {response}")
    
    return [f[0] for f in new_or_modified_files]

def load_file_hashes_from_supabase(supabase_client):
    """Load filenames and their hashes from Supabase."""
    response = supabase_client.table("file_hashes").select("*").execute()
    if response:
        logger.info("Successfully loaded file hashes from Supabase.")
        return {item["filename"]: item["hash"] for item in response.data}
    else:
        logger.error(f"Failed to load file hashes from Supabase. | Response: {response}")
        return {}
