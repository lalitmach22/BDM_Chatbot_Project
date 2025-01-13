import os, re
import json
import numpy as np
from datetime import datetime, timedelta
import pytz
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
#from sklearn.metrics.pairwise import cosine_similarity
from .extract_texts import logger
from .tokens import count_tokens, count_tokens_in_chat_history
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store_1 = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
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

def get_chat_history_from_supabase(supabase):
    """Retrieve all chat history from Supabase and return as a list of (question, answer) tuples."""
    try:
        response = supabase.table("chat_sessions_2").select("question, answer").execute()
        if response.data and isinstance(response.data, list):
            chat_history = [(entry["question"], entry["answer"]) for entry in response.data]
            logger.info(f"Retrieved {len(chat_history)} chat history entries from Supabase.")
            return chat_history
        else:
            logger.warning("No chat history found in Supabase or data is not in expected format.")
            return []
    except Exception as e:
        logger.error(f"Exception occurred while retrieving chat history from Supabase: {e}")
        return []

def save_chat_history_to_local(path, chat_history):
    """Save updated chat history to local file."""
    try:
        with open(path, 'w') as f:
            json.dump(chat_history, f)
        logger.info("Chat history saved to local storage.")
    except Exception as e:
        logger.error(f"Error saving chat history to local file: {e}")
def load_chat_history_from_local(path):
    """Load chat history from local file if it exists, otherwise return empty list."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                chat_history = json.load(f)
                logger.info(f"Loaded {len(chat_history)} chat history entries from local storage.")
                return chat_history
        except Exception as e:
            logger.error(f"Error loading chat history from local file: {e}")
            return []
    else:
        logger.info("No local chat history found. Starting a new session.")
        return []

def get_limited_chat_history(chat_history, limit=5):
    """Limit chat history to the last 'limit' number of entries."""
    return chat_history[-limit:]


def find_similar_question_faiss(user_input, vector_store_1, embeddings, k = 4, 
                                fetch_k = 5, lambda_mult = 0.5, filter = None,  
                                similarity_threshold = 0.9):
    """Find similar questions using FAISS-based vector search with maximal marginal relevance and score filtering."""
    user_input_embedding = embeddings.embed_query(user_input)
    results_with_scores = vector_store_1.max_marginal_relevance_search_with_score_by_vector(
        embedding=user_input_embedding,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        filter=filter
    )
    filtered_results = [
        (result[0].page_content.split("Answer:")[1].strip(), result[1])
        for result in results_with_scores
        if result[1] >= similarity_threshold
    ]
    
    if filtered_results:
        logger.info(f"Found {len(filtered_results)} similar questions with similarity score >= {similarity_threshold}.")
        best_match = filtered_results  
        logger.info(f"Found best similar answer with similarity score: {best_match}")
        return best_match
    else:
        logger.info(f"No similar questions found with similarity score >= {similarity_threshold}.")
        return None 

def process_user_input(supabase, retrieval_chain, email, name, user_input, chat_history, start_time=None):
    """Process the user's input and return the chatbot's response."""
    logger.info(f"Processing user input: {user_input} and type of user input is {type(user_input)}")      
    if start_time is None:
        start_time = datetime.now()

    current_time = datetime.now()
    elapsed_time = current_time - start_time  
    documents = [
        Document(
            page_content=f"Question: {question}\nAnswer: {answer}",
            metadata={"source": "chat_history"}
        )
        for question, answer in chat_history
    ]
    logger.info(f"Created  documents type is{type(documents)}")
      
    uuids = [str(uuid4()) for _ in range(len(documents))]
    try:
        vector_store_1.add_documents(documents=documents, ids=uuids)
        logger.info(f"Documents added to vector store. and lenght of documents added is {len(documents)}")
        logger.info(f"Vector store contains {vector_store_1.index.ntotal} vectors.")
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {e}")
        return "An error occurred while processing your input.", 0
    results = find_similar_question_faiss(user_input, vector_store_1, embeddings, k = 1, 
                                fetch_k = 5, lambda_mult = 0.5, filter = None,  
                                similarity_threshold = 0.95)
    if results:    
        logger.info(f"Answer to similar question is {results}and type is {type(results)} and number of similar questions are {len(results)}")
    
    if results and len(chat_history )> 50:
        logger.info(f"Found answer to similar question: {results}")
        return results, 0
    vector_store_1.delete(ids=uuids)  
    try:
        limited_chat_history = get_limited_chat_history(chat_history, limit=5)

        limited_chat_history_tuples = [tuple(pair) for pair in limited_chat_history]
        tokens_count = count_tokens_in_chat_history(limited_chat_history_tuples)
        response = retrieval_chain.invoke({"question": user_input, "chat_history": limited_chat_history_tuples})
        
        logger.info(f"Response type is {type(response)}")
        answer = response["answer"]
        chat_history.append((user_input, answer))
        save_chat_history_to_local(r'D:\BDM2\backend\app\chat_history.json', chat_history)            
        logger.info(f"Chatbot response: {answer}")
        tokens_count += count_tokens(user_input)
        logger.info(f"Number of tokens sent to API: {tokens_count}")

        if not user_input.lower() == "stop" or elapsed_time > timedelta(minutes=30):
            return answer , tokens_count
        if user_input.lower() == "stop" or elapsed_time > timedelta(minutes=30):
            save_session_to_supabase(supabase, email, name, chat_history)
            logger.info("Session data successfully saved to Supabase.")
            end_message = "Session data successfully saved. Please refresh to start a new session."
            chat_history.clear()
            logger.info("Chat history cleared after session end.")
            return end_message, tokens_count  
    except Exception as e:
        logger.error(f"Exception occurred while processing user input: {e}")
        return "An error occurred while processing your input. Please try again.", 0