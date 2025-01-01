import os

def store_embeddings(supabase, embedding_data):
    """
    Store embeddings in Supabase with a check for duplicates.
    :param embedding_data: List of dictionaries with text and embeddings
    """
    try:
        # Check if embeddings already exist (assuming embeddings are unique based on some property)
        for item in embedding_data:
            existing = supabase.table("embeddings").select("id").eq("text", item["text"]).execute()
            if existing.data:
                print(f"Embedding for text '{item['text']}' already exists, skipping.")
                continue
        
        response = supabase.table("embeddings").insert(embedding_data).execute()
        if response.error:
            print(f"Error storing embeddings: {response.error}")
        else:
            print("Embeddings stored successfully.")
    except Exception as e:
        print(f"Error interacting with Supabase: {e}")


def retrieve_relevant_texts(supabase, query_embedding, top_k=5):
    """
    Retrieve the most relevant texts based on a query embedding.
    :param query_embedding: Query embedding to compare
    :param top_k: Number of top results to retrieve
    :return: List of relevant text documents
    """
    try:
        response = supabase.rpc("match_embeddings", {
            "query_embedding": query_embedding,
            "top_k": top_k
        }).execute()
        if response.error:
            print(f"Error retrieving texts: {response.error}")
        else:
            return response.data
    except Exception as e:
        print(f"Error interacting with Supabase: {e}")
        return []

def store_chat_history(supabase, chat_id, chat_history):
    """
    Store chat history in Supabase.
    :param chat_id: Unique ID for the chat session
    :param chat_history: List of messages in the chat
    """
    try:
        response = supabase.table("chat_history").insert({
            "chat_id": chat_id,
            "history": chat_history
        }).execute()
        if response.error:
            print(f"Error storing chat history: {response.error}")
        else:
            print("Chat history stored successfully.")
    except Exception as e:
        print(f"Error interacting with Supabase: {e}")

def retrieve_chat_history(supabase, chat_id):
    """
    Retrieve chat history from Supabase.
    :param chat_id: Unique ID for the chat session
    :return: List of chat messages
    """
    try:
        response = supabase.table("chat_history").select("history").eq("chat_id", chat_id).execute()
        if response.error:
            print(f"Error retrieving chat history: {response.error}")
            return []
        elif response.data:
            # Extract and return the chat history
            return response.data[0]["history"]
        else:
            print("No chat history found for the given chat_id.")
            return []
    except Exception as e:
        print(f"Error interacting with Supabase: {e}")
        return []
