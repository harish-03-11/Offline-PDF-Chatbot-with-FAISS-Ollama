# process_pdf.py
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# --- Configuration ---
PDF_DIR = "data/pdfs"                   # Folder where your PDFs are stored
INDEX_FILE = "embeddings/faiss_index.pkl"  # Output file for the FAISS index & metadata
MODEL_NAME = "all-MiniLM-L6-v2"           # Sentence Transformer model for embeddings

# --- Step 1: Extract text from a PDF ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Step 2: Chunk text using LangChain ---
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

# --- Step 3: Embed text chunks ---
def embed_chunks(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# --- Step 4: Build FAISS index ---
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# --- Main Process ---
def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    metadata = []  # To store source info for each chunk
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.extend([{"source": pdf_file, "chunk_index": i} for i in range(len(chunks))])
    
    print("Embedding text chunks...")
    embeddings = embed_chunks(all_chunks, model)
    embeddings = np.array(embeddings).astype("float32")
    
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    # Save the FAISS index along with metadata and text chunks for later use
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"index": index, "metadata": metadata, "chunks": all_chunks}, f)
    print("FAISS index saved successfully to", INDEX_FILE)

if __name__ == "__main__":
    main()
