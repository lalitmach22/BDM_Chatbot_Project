import uuid
from datetime import datetime
from app.data import store_chat_history, store_embeddings, retrieve_relevant_texts
from app.vector_store import reload_vector_store_if_needed
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from supabase import create_client
import os
import Flask
from Flask import request, jsonify

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Check if the environment variables are set
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and key are not set in the environment variables.")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load the model and retrieval chain
prompt = """You are an intelligent and helpful assistant specializing in providing detailed, accurate, and practical answers for queries related to the BDM project. Your role is to:
    1. Guide users for the BDM project like formats of proposal, mide term report and final report. Page , font, line spacing settings
    2. Guide users in solving problems related to supply chain management, financial analysis, operational efficiency, and data-driven decision-making.
    3. Provide concise yet comprehensive explanations, including relevant examples, where necessary.
    4. If a user’s query is unclear, politely ask for clarification.
    5. If the question is outside your expertise or related to specific project data you don't have, inform the user and suggest alternative approaches or resources.
    6. Use a professional and approachable tone suitable for business and academic environments.
    Always ensure that your answers are actionable, focused, and aligned with best practices in business analysis and decision modeling."""


def load_model():
    return ChatGroq(
        temperature=0.8, 
        model="llama3-8b-8192",
        system_prompt=prompt
    )

# Initialize model, vector store, and retrieval chain
model = load_model()
vector_store = reload_vector_store_if_needed()
retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

# Function to simulate chatbot response with retrieval chain
def generate_chatbot_response(user_message, chat_history):
    """
    Generate a response for the chatbot using a retrieval chain.
    :param user_message: Message from the user
    :param chat_history: History of the current chat
    :return: A response message from the chatbot
    """
    # Use retrieval chain to get a response from the model
    response = retrieval_chain.invoke({
        "question": user_message,
        "chat_history": chat_history
    })
    answer = response["answer"]  # Extract the generated response
    return answer

def start_new_chat():
    """
    Start a new chat session.
    :return: A unique chat_id
    """
    chat_id = str(uuid.uuid4())  # Generate a unique ID for the chat session
    print(f"New chat session started with ID: {chat_id}")
    return chat_id

def process_user_message(chat_id, user_message, chat_history):
    """
    Process a user's message during a chat session.
    :param chat_id: Unique ID for the chat session
    :param user_message: Message from the user
    :param chat_history: History of the current chat
    :return: A response from the chatbot
    """
    # Generate the chatbot response using the retrieval chain
    bot_response = generate_chatbot_response(user_message, chat_history)

    # Format the chat history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_history.append({
        "timestamp": timestamp,
        "user_message": user_message,
        "bot_response": bot_response
    })

    # Store the updated chat history in the database (Supabase)
    store_chat_history(chat_id, chat_history)

    # Return the chatbot response and the updated chat history
    return bot_response, chat_history

# Initialize Flask app
app = Flask(__name__)

# Route for starting a new chat
@app.route('/start_chat', methods=['GET'])
def start_chat():
    chat_id = start_new_chat()
    return jsonify({"chat_id": chat_id}), 200

# Route to process user messages
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    chat_id = data.get("chat_id")
    user_message = data.get("user_message")
    
    if not chat_id or not user_message:
        return jsonify({"error": "chat_id and user_message are required"}), 400

    chat_history = []  # You could retrieve chat history from the database if needed
    bot_response, updated_chat_history = process_user_message(chat_id, user_message, chat_history)

    return jsonify({
        "bot_response": bot_response,
        "chat_history": updated_chat_history
    }), 200

# Main entry point to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
