import os, re
import json
import numpy as np
from datetime import datetime, timedelta
import pytz
import redis
#from sklearn.metrics.pairwise import cosine_similarity
from .extract_texts import logger
from .tokens import count_tokens

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Local cache for chat history
local_cache = {}
# Email validation regex
email_regex = re.compile(r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$")

def is_valid_email(email):
    """Validate email format."""
    return email_regex.match(email) is not None or email == "nitin@ee.iitm.ac.in" or email == "lalitmach22@gmail.com"

def save_session_to_supabase(supabase, email, name, chat_history):
    """Save the chat session to Supabase."""
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M")
    
    data_list = []
    for question, answer in chat_history:
        data = {
            "email": email,
            "name": name if name else None,
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
        }
        data_list.append(data)
    try:
        response = supabase.table("chat_sessions_2").insert(data_list).execute()
        if response:
            logger.info("Session data saved to Supabase.")
            return True
        else:
            logger.error(f"Error saving session data to Supabase: {response}")
            return False
    except Exception as e:
        logger.error(f"Exception occurred while saving session data to Supabase: {e}")
        return False

def find_similar_question_faiss(user_input, vector_store, embedder, chat_history):
    """Find a similar question using FAISS-based vector search."""
    if not chat_history:
        return None, None
    try:
        # Generate embedding for the user input
        user_embedding = embedder.embed_query(user_input).reshape(1, -1)

        # Perform similarity search in the FAISS index (top 1 match)
        distances, indices = vector_store.search(user_embedding, k=1)

        # Check if a similar question is found and apply a threshold
        similarity_threshold = 0.8  # Adjust as needed
        if distances[0][0] < (1 - similarity_threshold):  # FAISS uses distance; lower is better
            similar_index = indices[0][0]
            similar_question, similar_answer = chat_history[similar_index]
            return similar_question, similar_answer
    except Exception as e:
        logger.error(f"Exception occurred while finding similar question: {e}")

    return None, None
def process_user_input(supabase, retrieval_chain, email, name, user_input, vector_store, embedder, chat_history=None, start_time=None):
    """Process the user's input and return the chatbot's response."""
    logger.info(f"Processing user input: {user_input}")

    if chat_history is None:
        chat_history = []

    if start_time is None:
        start_time = datetime.now()

    current_time = datetime.now()
    elapsed_time = current_time - start_time    

    # Check for similar questions using FAISS
    similar_question, similar_answer = find_similar_question_faiss(user_input, vector_store, embedder, chat_history)
    if similar_question:
        logger.info(f"Found similar question: {similar_question}")
        # Return similar answer with 0 token count (no API call)
        return similar_answer, 0

    try:
        # If no similar question, process the question and get the answer from the API
        response = retrieval_chain.invoke({"question": user_input, "chat_history": chat_history})
        answer = response["answer"]
        chat_history.append((user_input, answer))
        
        logger.info(f"Chatbot response: {answer}")
        tokens_count = count_tokens(user_input)
        logger.info(f"Number of tokens sent to API: {tokens_count}")
        
        # If the user types "stop" or the session has timed out, save the session
        if user_input.lower() == "stop" or elapsed_time > timedelta(minutes=30):  # Adjusted session timeout
            save_session_to_supabase(supabase, email, name, chat_history)
            logger.info("Session data successfully saved to Supabase. Please refresh to start a new session.")
            
            # Clear chat history after saving session data
            chat_history.clear()  # This will remove the chat history from memory
            logger.info(f"Chat history cleared after session end.")
            
            # Return the answer and token count even when session ends, to maintain consistency
            return "Session data successfully saved to Supabase. Please refresh to start a new session", tokens_count

        # Save updated chat history to local cache and Redis
        chat_history_key = f"chat_history:{email}"
        local_cache[email] = chat_history        
        redis_client.setex(chat_history_key, timedelta(days=7), json.dumps(chat_history))  # Set expiration of 7 days
        return answer, tokens_count
    except Exception as e:
        logger.error(f"Exception occurred while processing user input: {e}")
        return "An error occurred while processing your input. Please try again.", 0