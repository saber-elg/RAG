import os
import signal 
import sys

# Import required libraries for text generation and retrieval 
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Text Generation Engine API KEY (GEMINI)
GEMINI_API_KEY = "AIzaSyC7Go3tZ7qyXCVg0lN9g2WrOmCta2aLem4"

# Gracefully exits the program upon receiving a SIGINT (Ctrl+C) signal.
def signal_handler(sig,frame):
    print('\nThanks for using Gemini. ')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Creates a RAG (Retrieval-Augmented Generation) prompt suitable for Gemini.
def generate_rag_prompt(query,context):
    prompt= """ You are a  helpful and informative bot that answers questions using text from the referecence context included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all background information \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the context is irrelevant to the answer, you may answer it from your knowledge.  
    Question: {query}
    Context: {context}
    
    Answer:""".format(query=query, context=context)
    return prompt

# Retrieves relevant context from the database using a text embedding model.
def get_relevant_context_from_db(query):
    context = ""
    # Replace with your preferred text embedding model and vector store implementation
    embeddings_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_rag", embedding_function=embeddings_function)

    # Perform similarity search to find relevant context
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
        return context

# Generates an answer to the user's query using the Gemini generative model.
def generate_answer(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

welcome_text = generate_answer("Can you quickly introduce yourself ?")
print(welcome_text)

while True:
    print("----------------------------")
    print("what would you like to ask?")
    query=input("Query: ")

    # Retrieve relevant context from the database
    context = get_relevant_context_from_db(query)

    # Generate RAG prompt for Gemini
    prompt=generate_rag_prompt(query=query, context=context)

    # Generate answer using Gemini
    answer=generate_answer(prompt=prompt)
    print(answer)
