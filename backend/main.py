import os
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import sqlite3
from typing import List
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check if the API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")
api_key = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update this to match your frontend port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Initialize database
conn = sqlite3.connect('documents.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        upload_date DATETIME NOT NULL
    )
''')
conn.commit()

# Global variable to store the current index
current_index = None

class Question(BaseModel):
    question: str

class Document(BaseModel):
    id: int
    filename: str
    upload_date: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global current_index
    try:
        if file.filename.endswith(".pdf"):
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"File saved: {file_path}")
            
            # Store document information in the database
            cursor.execute('''
                INSERT INTO documents (filename, upload_date) VALUES (?, ?)
            ''', (file.filename, datetime.now().isoformat()))
            conn.commit()
            
            logger.info("Document info stored in database")
            
            # Create an index from the PDF using HuggingFace embeddings
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            current_index = Chroma.from_documents(texts, embeddings)
            
            logger.info("Index created successfully")
            
            return {"message": "File uploaded successfully"}
        else:
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

@app.post("/ask")
async def ask_question(question: Question):
    global current_index
    try:
        if current_index is None:
            raise HTTPException(status_code=400, detail="No PDF has been uploaded yet")
        
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=current_index.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        result = qa_chain({"query": question.question})
        return {"answer": result["result"]}
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the question: {str(e)}")

@app.get("/documents", response_model=List[Document])
async def get_documents():
    try:
        cursor.execute('SELECT * FROM documents ORDER BY upload_date DESC')
        documents = cursor.fetchall()
        # Log the fetched documents for debugging
        logger.info(f"Fetched documents: {documents}")
        return [Document(id=doc[0], filename=doc[1], upload_date=doc[2]) for doc in documents]
    except Exception as e:
        logger.error(f"Error in get_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)