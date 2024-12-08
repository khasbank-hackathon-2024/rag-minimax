import torch
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.generativeai import GenerativeModel
import os
from dotenv import load_dotenv
from functools import lru_cache
from datetime import datetime
from pydantic import BaseModel
import re
from langchain.schema import Document
from knowledge_base import KnowledgeBase
from typing import Dict, List
import json
from pathlib import Path
from create_vectorstore import load_knowledge_base, create_vector_store

load_dotenv()

def load_data(data_dir="data/"):
    """Load data and create knowledge base"""
    kb = KnowledgeBase()
    
    # News data
    news_data = _load_csv(f"{data_dir}khas_bank_news.csv", 
                         source_type="news")
    
    # Products data  
    product_data = _load_csv(f"{data_dir}khas_bank_products.csv",
                            source_type="product")
    
    # Pages data
    pages_data = _load_csv(f"{data_dir}khas_bank_pages.csv", 
                          source_type="page")

    # Add to knowledge base
    kb.add_documents(news_data + product_data + pages_data)
    
    return kb.data["documents"]

def _load_csv(filepath: str, source_type: str) -> List[Dict]:
    """Helper to load CSV files into knowledge base format"""
    loader = CSVLoader(
        file_path=filepath,
        source_column="link",
        metadata_columns=["title", "link"],
        content_columns=["content"]
    )
    documents = loader.load()[1:]
    
    # Convert to knowledge base format
    return [{
        "content": doc.page_content,
        "metadata": {
            **doc.metadata,
            "source_type": source_type
        }
    } for doc in documents]

def load_vector_store(load_path="faiss_index"):
    """Load existing vector store"""
    vector_store_path = Path(__file__).parent / load_path
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    return FAISS.load_local(
        str(vector_store_path), 
        embeddings, 
        allow_dangerous_deserialization=True
    )

def setup_rag_chain(vector_store, api_key):
    """Setup RAG chain with Gemini LLM"""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=1024,
        )

        # Setup retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
            }
        )


        prompt = ChatPromptTemplate.from_messages([
            ("human", """
            Та Хас Банкны албан ёсны дижитал туслах "Хас" бөгөөд дараах үндсэн зарчмуудыг баримтална:

            ХАРИУЛАХ АРГА БАРИЛ:
            1. Мэндчилгээ: "Сайн байна уу" гэж эхэлж, асуултын төрлөөс хамааран тохирох мэндчилгээ хэрэглэнэ
            2. Хэл найруулга: Албан ёсны, эелдэг боловч энгийн ойлгомжтой хэллэг ашиглана
            3. Хариултын бүтэц: 
               - Товч тодорхой, 2-3 өгүүлбэрт багтаана
               - Чухал тоон мэдээллийг тодруулж бичнэ
               - Шаардлагатай бол холбоос өгнө
            
            МЭДЭЭЛЛИЙН СТАНДАРТ:
            1. Зөвхөн баталгаат эх сурвалжийн мэдээлэл ашиглана
            2. Тоон мэдээллийг бүрэн, таслалтай бичнэ
            3. Хугацаа, дүн, нөхцөлийг тодорхой дурдана
            4. Эргэлзээтэй мэдээлэл байвал "Уучлаарай, энэ талаар баталгаат мэдээлэл байхгүй байна" гэж хариулна
            
            ХОРИГЛОХ ЗҮЙЛС:
            1. Таамаглал, төсөөлөл хэлэхгүй
            2. Худал, буруу мэдээлэл өгөхгүй
            3. Банкны нууцад хамаарах мэдээлэл дэлгэхгүй
            4. Хувь хүний мэдээлэл асуухгүй
            
            НЭМЭЛТ ҮЙЛДЛҮҮД:
            1. Хэрэв асуулт тодорхойгүй бол тодруулах асуулт тавина
            2. Шаардлагатай бол 1800-1888 утасны дугаарыг өгнө
            3. Онлайн үйлчилгээний хувьд аппликейшн татах заавар өгнө
            4. Салбарын мэдээлэл өгөхдөө ажиллах цагийг дурдана

            Дараах мэдээлэлд үндэслэн хариулна уу:
            {context}

            Асуулт: {input}
            """)
        ])

        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)
    except Exception as e:
        print(f"Error setting up RAG chain: {str(e)}")
        raise

def check_data_status():
    """Check data and vector store status"""
    try:
        # Check if vector store exists
        if os.path.exists("faiss_index"):
            print("\nVector store status:")
            print("✓ faiss_index folder exists")
            
            # Check data files
            data_files = [
                "../data/khas_bank_news.csv",
                "../data/khas_bank_products.csv",
                "../data/khas_bank_pages.csv"
            ]
            
            print("\nData files status:")
            for file in data_files:
                if os.path.exists(file):
                    size = os.path.getsize(file)
                    print(f"✓ {file.split('/')[-1]} ({size/1024:.1f} KB)")
                else:
                    print(f"✗ {file.split('/')[-1]} (missing)")
            
            return True
        else:
            print("✗ No vector store found. Data needs to be loaded.")
            return False
            
    except Exception as e:
        print(f"Error checking data status: {str(e)}")
        return False

class RAGBot:
    def __init__(self, api_key, vector_store_path="faiss_index", force_new=False):
        try:
            if force_new:
                raise Exception("Forcing new vector store creation")
                
            # Vector store path will be relative to src folder
            self.vector_store = load_vector_store(vector_store_path)
            print("✓ Loaded existing vector store")
        except Exception as e:
            print(f"Creating new vector store... Reason: {str(e)}")
            documents = load_knowledge_base()
            self.vector_store = create_vector_store(documents, vector_store_path)
            
        self.chain = setup_rag_chain(self.vector_store, api_key)

    # def check_data_status(self):
    #     """Check data and vector store status"""
    #     try:
    #         # Check if vector store exists
    #         if os.path.exists("faiss_index"):
    #             print("\nVector store status:")
    #             print("✓ faiss_index folder exists")

    #             # Check data files
    #             data_files = [
    #                 "../data/khas_bank_news.csv",
    #                 "../data/khas_bank_products.csv",
    #                 "../data/khas_bank_pages.csv"
    #             ]
                
    #             print("\nData files status:")
    #             for file in data_files:
    #                 if os.path.exists(file):
    #                     size = os.path.getsize(file)
    #                     print(f"✓ {file.split('/')[-1]} ({size/1024:.1f} KB)")
    #                 else:
    #                     print(f"✗ {file.split('/')[-1]} (missing)")
                
    #             return True
    #         else:
    #             print("✗ No vector store found. Data needs to be loaded.")
    #             return False
                
    #     except Exception as e:
    #         print(f"Error checking data status: {str(e)}")
    #         return False

    def ask(self, question):
        try:
            
            # word_count = len(question.split())
            # print(f"Word Count: {word_count} words")
            response = self.chain.invoke({"input": question})
            if not response or not response.get("answer"):
                return "Уучлаарай, хариулт олдсонгүй." 
            
            # print(f"User Query: {question}")
            # print(f"Augmented Context: {response.get('context', 'No context found')}")
            return self.post_process_answer(response["answer"])
        except Exception as e:
            return f"Алдаа гарлаа: {str(e)}"

    def post_process_answer(self, answer):
        """Хариултыг боловсруулах"""
        # Эх сурвалж нэмэх
        if "source" in answer:
            answer += f"\nЭх сурвалж: {answer['source']}"
        
        
        return answer

    @lru_cache(maxsize=1000)
    def cached_ask(self, question):
        return self.ask(question)

    def update_knowledge_base(self):
        """Шинэ мэдээлэл нэмэх"""
        new_docs = fetch_new_documents()
        if new_docs:
            self.vector_store.add_documents(new_docs)

def log_interaction(question, answer, feedback=None):
    """Хэрэглэгчийн харилцааг хадгалах"""
    log_entry = {
        "timestamp": datetime.now(),
        "question": question,
        "answer": answer,
        "feedback": feedback
    }
    save_to_log(log_entry)

def main():
    # check_data_status()
    
    # Replace with your API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        api_key = input("Please enter your Google API key: ")
    
    try:
        # Initialize RAG bot
        bot = RAGBot(api_key)
        
        # Interactive question-answering loop
        print("RAG Bot ready! Type 'quit' to exit")
        while True:
            question = input("\nYour question: ")
            if question.lower() == 'quit':
                break
            
            answer = bot.ask(question)
            print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_rag_system():
    """Unit tests for RAG system"""
    test_cases = [
        ("IPO хэзээ хийсэн бэ?", "contains_date"),
        ("Хаяг хааyа вэ?", "contains_address"),
        ("Invalid question", "error_handling")
    ]
    run_tests(test_cases)

def clean_text(text):
    """Clean and preprocesstext"""
    if isinstance(text, Document):
        text = text.page_content
        
    # Remove special characters and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,\-:;]', ' ', text)
    
    # Truncate if too long (Gemini has token limits)
    max_chars = 2048
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return text.strip()

if __name__ == "__main__":
    main()

