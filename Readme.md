Download dcocuments from link "https://drive.google.com/drive/folders/1ncoIEIw_P2__rQHBdbyLjkQbOHNMmQi6?usp=sharing" in the hidden_docs folder in backend

Inside backend create a file .env and keep supabase url, supabase key and groq api key inside it. You can obtaing supabase url and key by mailing to lalitmach22@gmail.com. Else you can make your owr own databases in supabase and use url and key.

Use your own groqapi key.


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


    elif not vector_store:
        # No new files and no vector store - build a vector store from all documents
        logger.info("No new files and no existing vector store. Building new vector store from all documents.")
        document_texts = load_hidden_documents(directory)
        if not document_texts:
            logger.warning("No documents found in the directory to build a vector store.")
            return None
        vector_store = create_vector_store(document_texts)
        vector_store.save_local(VECTOR_STORE_PATH)
        logger.info("New vector store created and saved.")

    else:
        # No new files, but an existing vector store is found
        logger.info("No new files. Using existing vector store.")

    return vector_store