import os

# Prevent HuggingFace tokenizers from deadlocking the Streamlit server
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable ChromaDB telemetry to prevent network hanging
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Set HF cache to local directory so Render preserves it between build and deploy
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.huggingface_cache')

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

def main():
    print("Loading documents...")
    # Map file extensions to loaders
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }
    
    docs = []
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext in loaders:
            try:
                loader = loaders[ext](filepath)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                
    if not docs:
        print("No documents found in data directory.")
        return

    print(f"Loaded {len(docs)} documents.")
    
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    print("Generating embeddings and saving to Chroma...")
    # Use a local, lightweight embedding model to save API calls
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the vector store
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    
    # Persist the database
    db.persist()
    print(f"Vector DB successfully created at {CHROMA_PATH}")

if __name__ == "__main__":
    main()
