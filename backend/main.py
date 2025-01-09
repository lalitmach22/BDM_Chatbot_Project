import os
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_groq import ChatGroq
from app.vector_store import reload_vector_store_if_needed, embedder
from app.chat import is_valid_email, process_user_input
from langchain.chains import ConversationalRetrievalChain
from app.extract_texts import logger,load_hidden_documents
from app.embeddings import store_embeddings_in_supabase



directory = "hidden_docs"
# Load environment variables from .env file
load_dotenv()

# Supabase credentials
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = None

def load_model():
    global model
    if model is None:
        try:
            model = ChatGroq(temperature=0.8, model="llama3-8b-8192")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError("Failed to load model.")
    return model
    
model = load_model()

supabase: Client = create_client(url, key)

clean_texts = load_hidden_documents(directory)
store_embeddings_in_supabase(supabase, clean_texts, embedder)
logger.info(f"Embeddings stored in supabase")

# Reload vector store if needed
vector_store = reload_vector_store_if_needed(directory, supabase)
if vector_store is None:
    raise ValueError("Failed to initialize vector_store. Ensure hidden_docs folder and embeddings setup are correct.")

# Create retrieval chain
retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())
# Initialize Flask app
app = Flask(__name__)

@app.route('/validate_email', methods=['POST'])
def validate_email():
    try:
        email = request.form['email']
        if is_valid_email(email):
            logger.info(f"Email validated: {email}")
            return jsonify({"status": "success", "message": "Email validated successfully!"})
        else:
            logger.warning(f"Invalid email format: {email}")
            return jsonify({"status": "error", "message": "Invalid email format."})
    except Exception as e:
        logger.error(f"Error in validate_email: {e}")
        return jsonify({"status": "error", "message": "An error occurred during email validation."})

@app.route('/chat', methods=['POST'])
def ask_question():
    try:
        email = request.json['email']  
        name = request.json.get('name', '')  # Default name as empty string if not provided
        user_input = request.json['question']
        chat_history = request.json.get('chat_history', [])
        start_time = request.json.get('start_time', datetime.now().isoformat())
        
        logger.info(f"Received question: {user_input} from {email}")
        
        # Call the process_user_input function
        answer, tokens_count = process_user_input(supabase, retrieval_chain, email, name, user_input, chat_history, datetime.fromisoformat(start_time))
        
        logger.info(f"Question processed: {user_input}")
        
        # Include chat_history in the response for debugging
        return jsonify({
            "status": "success", 
            "answer": answer, 
            "tokens_count": tokens_count,
            "chat_history": chat_history  # Add chat_history here to check the chat history
        })
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return jsonify({
            "status": "error", 
            "message": "An error occurred while processing the question."
        })
    
@app.route('/get_token_count_from_input', methods=['POST'])
def get_token_count_from_input():
    try:
        email = request.json['email']
        name = request.json.get('name', '')
        user_input = request.json['question']
        
        # Omit start_time unless session is being considered
        _, tokens_count = process_user_input(supabase, retrieval_chain, email, name, user_input)
        
        logger.info(f"Token count for user input '{user_input}': {tokens_count}")
        return jsonify({"status": "success", "token_count": tokens_count})
    
    except Exception as e:
        logger.error(f"Error in get_token_count_from_input: {e}")
        return jsonify({"status": "error", "message": "An error occurred while counting tokens."})

if __name__ == '__main__':
    logger.info("Starting app...")
    app.run(debug=True)