# create_vectorstore.py 
import shutil
import os
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import json

def load_knowledge_base(data_dir: str = "data") -> List[Document]:
    """Load documents from knowledge base"""
    project_root = Path(__file__).parent.parent
    kb_path = project_root / data_dir / "knowledge_base.json"
    
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)
    
    documents = []
    for doc in kb["documents"]:
        documents.append(
            Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )
        )
    
    print(f"Loaded {len(documents)} documents from knowledge base")
    return documents

def create_vector_store(documents: List[Document], embeddings_dir: str = "embeddings"):
    """Create vector store from documents"""
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    
    # Create and save vector store
    project_root = Path(__file__).parent.parent
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store_path = Path(__file__).parent / embeddings_dir
    vector_store.save_local(str(vector_store_path))
    
    print(f"Created vector store with {len(splits)} vectors in {vector_store_path}")
    return vector_store

# Main execution
if __name__ == "__main__":
    # 1. Delete existing vector store if it exists
    faiss_path = Path(__file__).parent / "faiss_index"
    if faiss_path.exists():
        shutil.rmtree(faiss_path)
        print(f"✓ Removed old vector store from {faiss_path}")

    # 2. Create new vector store
    documents = load_knowledge_base()  # Load your documents
    vector_store = create_vector_store(documents, "faiss_index")
    print("✓ Created new vector store with updated embeddings model")

