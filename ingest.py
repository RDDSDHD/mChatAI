from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Load OpenAI API key
api_key = "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"

# Load the Notion content located in the folder 'notion_content'
loader = PyPDFLoader("/home/sk30613/PycharmProjects/RDchatbot/input_files/internal_knowledge_combined.pdf")
pages = loader.load()

# Split the Notion content into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Convert all chunks into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save to local folder 'faiss_index'
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print('Local FAISS index has been successfully saved.')
