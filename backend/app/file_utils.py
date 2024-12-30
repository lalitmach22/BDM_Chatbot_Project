import os, re, mimetypes
from docx import Document
import pandas as pd
from pptx import Presentation
from langchain_community.document_loaders import PyPDFLoader

def clean_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Fix words broken by line breaks or formatting
    # Matches a lowercase/uppercase word followed by a line break without a space
    text = re.sub(r'(?<=[a-zA-Z])(?=[A-Z])', ' ', text)  # Add space between words when they are stuck together    
    # Fix punctuation followed by words with no space
    text = re.sub(r'(?<=[.,?!;])(?=[a-zA-Z])', ' ', text)  # Add space after punctuation if missing
    # Standardize newlines for better formatting
    text = re.sub(r'\.\s+', '.\n', text)  # Add newlines after sentences
    text = re.sub(r'(?<=:)\s+', '\n', text)  # Add newlines after colons
    # Additional cleanup (if needed)    
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Helper functions to load different types of documents in chunks
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    all_texts = []
    pages = loader.load_and_split()
    
    # Process each page one by one to save memory
    for page in pages:
        all_texts.append(page.page_content)
    return all_texts

def load_word(file_path, chunk_size=10):
    doc = Document(file_path)
    all_texts = []
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]

    # Break paragraphs into chunks
    for i in range(0, len(paragraphs), chunk_size):
        chunk = "\n".join(paragraphs[i:i+chunk_size])
        all_texts.append(chunk)
        
    return all_texts

def load_text(file_path, chunk_size=1024):
    with open(file_path, "r", encoding="utf-8") as file:
        all_texts = []
        while chunk := file.read(chunk_size):
            all_texts.append(chunk)
    return all_texts

def load_excel(file_path, chunk_size=1000):
    excel_data = pd.read_excel(file_path, chunksize=chunk_size)
    all_texts = []
    
    for chunk in excel_data:
        all_texts.append(chunk.to_string(index=False))
        
    return all_texts

def load_csv(file_path, chunk_size=1000):
    csv_data = pd.read_csv(file_path, chunksize=chunk_size)
    all_texts = []
    
    for chunk in csv_data:
        all_texts.append(chunk.to_string(index=False))
        
    return all_texts

def load_pptx(file_path):
    presentation = Presentation(file_path)
    all_texts = []
    
    # Extract text from slides
    for slide in presentation.slides:
        slide_text = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                slide_text.append(shape.text)
        all_texts.append("\n".join(slide_text))
    
    return all_texts

def load_hidden_documents(directory="hidden_docs"):
    """Load all supported file types from a directory and return their content in chunks."""
    all_texts = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        mime_type, _ = mimetypes.guess_type(file_path)

        # Handle PDF files
        if filename.endswith(".pdf"):
            all_texts.extend(load_pdf(file_path))

        # Handle Word files (.docx)
        elif filename.endswith(".docx"):
            all_texts.extend(load_word(file_path))

        # Handle Text files (.txt)
        elif filename.endswith(".txt"):
            all_texts.extend(load_text(file_path))

        # Handle Excel files (.xlsx and .xls)
        elif filename.endswith(('.xlsx', '.xls')):
            all_texts.extend(load_excel(file_path))

        # Handle CSV files (.csv)
        elif filename.endswith(".csv"):
            all_texts.extend(load_csv(file_path))

        # Handle PowerPoint files (.pptx)
        elif filename.endswith(".pptx"):
            all_texts.extend(load_pptx(file_path))

    # Clean the collected texts
    cleaned_texts = [clean_text(text) for text in all_texts]
    return cleaned_texts
