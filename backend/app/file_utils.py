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
    # Additional cleanup (if needed)    text = text.strip()  # Remove leading and trailing whitespace
    return text

def load_hidden_documents(directory="hidden_docs"):
    """Load all supported file types from a directory and return their content."""
    all_texts = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        mime_type, _ = mimetypes.guess_type(file_path)

        # Handle PDF files
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            all_texts.extend([page.page_content for page in pages])

        # Handle Word files (.docx)
        elif filename.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            all_texts.append(text)

        # Handle Text files (.txt)
        elif filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                all_texts.append(file.read())

        # Handle Excel files (.xlsx and .xls)
        elif filename.endswith(('.xlsx', '.xls')):
            excel_data = pd.read_excel(file_path)
            text = excel_data.to_string(index=False)
            all_texts.append(text)

        # Handle CSV files (.csv)
        elif filename.endswith(".csv"):
            csv_data = pd.read_csv(file_path)
            text = csv_data.to_string(index=False)
            all_texts.append(text)

        # Handle PowerPoint files (.pptx)
        elif filename.endswith(".pptx"):
            presentation = Presentation(file_path)
            for slide in presentation.slides:
                slide_text = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        slide_text.append(shape.text)
                all_texts.append("\n".join(slide_text))

    cleaned_texts = [clean_text(text) for text in all_texts]
    return cleaned_texts