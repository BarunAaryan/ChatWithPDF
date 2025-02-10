import os
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
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
from apscheduler.schedulers.background import BackgroundScheduler
import shutil
import time
from uuid import uuid4

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
        upload_date DATETIME NOT NULL,
        user_id TEXT NOT NULL
    )
''')
conn.commit()

# Add user_id column if it doesn't exist
cursor.execute("PRAGMA table_info(documents)")
columns = [info[1] for info in cursor.fetchall()]
if 'user_id' not in columns:
    cursor.execute("ALTER TABLE documents ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
    conn.commit()

# Global variable to store the current index
current_index = None

# Function to delete files older than 24 hours
def cleanup_uploads():
    now = time.time()
    uploads_dir = "uploads"
    for user_dir in os.listdir(uploads_dir):
        user_path = os.path.join(uploads_dir, user_dir)
        if os.path.isdir(user_path):
            for filename in os.listdir(user_path):
                file_path = os.path.join(user_path, filename)
                if os.path.isfile(file_path):
                    file_age = now - os.path.getmtime(file_path)
                    if file_age > 24 * 3600:  # 24 hours
                        os.remove(file_path)
                        logger.info(f"Deleted old file: {file_path}")

# Start the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_uploads, 'interval', hours=1)
scheduler.start()

# Shut down the scheduler when exiting the app
import atexit
atexit.register(lambda: scheduler.shutdown())

class Question(BaseModel):
    question: str

class Document(BaseModel):
    id: int
    filename: str
    upload_date: str

def get_user_id():
    # Generate a unique user ID for each session
    return str(uuid4())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Depends(get_user_id)):
    global current_index
    try:
        user_upload_dir = os.path.join("uploads", user_id)
        os.makedirs(user_upload_dir, exist_ok=True)

        if file.filename.endswith(".pdf"):
            file_path = os.path.join(user_upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"File saved: {file_path}")
            
            # Store document information in the database
            cursor.execute('''
                INSERT INTO documents (filename, upload_date, user_id) VALUES (?, ?, ?)
            ''', (file.filename, datetime.now().isoformat(), user_id))
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
            
            return {"message": "File uploaded successfully", "user_id": user_id}
        else:
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

@app.post("/ask")
async def ask_question(question: Question, user_id: str = Depends(get_user_id)):
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
async def get_documents(user_id: str = Depends(get_user_id)):
    try:
        cursor.execute('SELECT * FROM documents WHERE user_id = ? ORDER BY upload_date DESC', (user_id,))
        documents = cursor.fetchall()
        # Log the fetched documents for debugging
        logger.info(f"Fetched documents: {documents}")
        return [Document(id=doc[0], filename=doc[1], upload_date=doc[2]) for doc in documents]
    except Exception as e:
        logger.error(f"Error in get_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)