import logging
from flask import Flask, request, jsonify
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from flask_cors import CORS
import os
import requests
from bson import ObjectId

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

client = MongoClient(os.getenv("MONGO_URI"))
db = client['test']
courses_collection = db['courses']

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings()
faiss_index_path = "faiss_index"

def get_pdf_text(pdf_url):
    logger.info(f"Fetching PDF from URL: {pdf_url}")
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        text = ""
        with pdfplumber.open("temp.pdf") as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        logger.info("PDF text successfully extracted.")
        return text
    else:
        logger.error(f"Failed to fetch PDF, status code: {response.status_code}")
        return ""

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def process_pdf_and_create_vectorstore(pdf_url):
    text = get_pdf_text(pdf_url)
    if text:
        text_chunks = get_text_chunks(text)
        metadatas = [{"source": "course_pdf"}] * len(text_chunks)
        vectorstore = FAISS.from_texts(text_chunks, embeddings, metadatas)
        vectorstore.save_local(faiss_index_path)
        logger.info("Vectorstore created and saved successfully.")
        return vectorstore
    else:
        logger.error("No text extracted from PDF, vectorstore cannot be created.")
        return None

def get_retrieval_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

def generate_mcq_options(question_text):
    prompt = f"Generate four answer choices for the question: '{question_text}'. Ensure one answer is correct and the other three are plausible but incorrect."
    try:
        response = llm.invoke(prompt, max_tokens=100)
        generated_text = response.content
        options = generated_text.strip().split('\n')
        correct_answer = options[0] if options else None
        return options, correct_answer
    except Exception as e:
        logger.error(f"Error generating MCQ options: {e}")
        return [], None

@app.route('/chat/<course_id>', methods=['POST'])
def chat_with_course_pdf(course_id):
    data = request.json
    question = data.get('question')
    
    logger.info(f"Received question: {question}")
    course = courses_collection.find_one({"_id": ObjectId(course_id)})
    if not course:
        logger.error("Course not found.")
        return jsonify({"error": "Course not found"}), 404

    pdf_url = course.get("pdfUrl")
    if pdf_url:
        vectorstore = process_pdf_and_create_vectorstore(pdf_url)
        if vectorstore:
            retrieval_chain = get_retrieval_chain(vectorstore)
            response = retrieval_chain.invoke({"query": question})

            if response and 'result' in response:
                logger.info(f"Generated response successfully: {response['result']}")
                
                raw_questions = response['result'].split('\n')
                mcqs = []
                for raw_question in raw_questions:
                    if raw_question.strip():
                        options, correct_answer = generate_mcq_options(raw_question.strip())
                        mcqs.append({
                            "question": raw_question.strip(),
                            "choices": options,
                            "answer": correct_answer
                        })

                # Check if MCQs generated, fallback to default if empty
                if not mcqs:
                    mcqs = [{"question": "Sample question?", "choices": ["A", "B", "C", "D"], "answer": "A"}]

                return jsonify({
                    "mcqs": mcqs,
                })
            else:
                logger.error("Failed to generate a valid response.")
                return jsonify({"error": "Failed to generate response"}), 500
        else:
            logger.error("Failed to process the PDF.")
            return jsonify({"error": "Failed to process the PDF"}), 500
    else:
        logger.error("No PDF found for this course.")
        return jsonify({"error": "No PDF found for this course"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5003)
