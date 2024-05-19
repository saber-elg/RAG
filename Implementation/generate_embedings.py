from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF document, it can be more than one document.
loaders = [PyPDFLoader('./test_samples/monopoly.pdf')]

# Initializes an empty list to store the extracted document content
docs = []

# Loops through each document loader and appends it to the docs list the extracted
# text content from the PDF generated with the load() method on the PyPDFLoader.
for file in loaders:
    docs.extend(file.load())

# Split text into chunks (optional) to optimise the data access.
text_splitter =RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
docs =text_splitter.split_documents(docs)
# Generate text embeddings for each document chunk
embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})

# Creates a Chroma vector store by processing each document chunk (docs) through the embedding 
# function and storing the resulting embeddings along with the document information. 
# The persist_directory argument specifies the location to store the vector data persistently.
vectorstore = Chroma.from_documents(docs, embeddings_function, persist_directory= "./chroma_db_rag")

print(f"Number of documents stored in the vectorstore: {vectorstore._collection.count()}")