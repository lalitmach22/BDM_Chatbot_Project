import uuid
from data import store_chat_history, retrieve_relevant_texts
from datetime import datetime
from vector_store import *
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS


# Load the model and retrieval chain
def load_model():
    return ChatGroq(temperature=0.8, model="llama3-8b-8192")

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


if __name__ == "__main__":
    # Example of starting a new chat and processing a user message
    chat_id = start_new_chat()
    chat_history = []  # Initialize an empty chat history

    user_message = "What is machine learning?"
    bot_response, chat_history = process_user_message(chat_id, user_message, chat_history)

    print(f"Bot Response: {bot_response}")
    print(f"Updated Chat History: {chat_history}")
