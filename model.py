import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from supabase import create_client

# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.environ.get("supabase_url")
SUPABASE_KEY = os.environ.get("supabase_key")
GEMINI_API_KEY = os.environ.get("gemini_api_key")

if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
    print("[ERROR] One or more environment variables are missing!")
    exit(1)

# --- 1. Load documents ---
def load_docs():
    docs = []
    docs_folder = "docs"
    if not os.path.exists(docs_folder):
        print(f"[ERROR] Docs folder '{docs_folder}' not found!")
        return docs

    for file in os.listdir(docs_folder):
        path = os.path.join(docs_folder, file)
        if file.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    return docs

# --- 2. Split into chunks ---
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Created {len(chunks)} chunks")
    return chunks

# --- 3. Store embeddings in Supabase ---
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    vectorstore = SupabaseVectorStore(
        client=client,
        table_name="learning_chunks",
        embedding=embeddings,
        query_name="match_learning_chunks"
    )

    docs_to_store = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk.page_content,
            metadata={
                "role": "student",
                "difficulty": "easy",
                "format": "pdf",
                "topic": "general",
                "level": 1,
                "progress_level": 0,
                "source": chunk.metadata.get("source", "unknown")
            }
        )
        docs_to_store.append(doc)

    vectorstore.add_documents(docs_to_store)
    print(f"[INFO] Stored {len(docs_to_store)} chunks in Supabase")
    return vectorstore

# --- 4. Build QA pipeline ---
def build_qa(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ✅ Using valid Gemini model
    # Try these in order: gemini-pro, gemini-1.5-flash, gemini-1.5-pro
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Start with the most compatible version
        temperature=0,
        google_api_key=GEMINI_API_KEY  # Note: use google_api_key parameter
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa

# --- 5. Run chatbot ---
if __name__ == "__main__":
    print("[INFO] Loading documents...")
    docs = load_docs()
    if not docs:
        print("[ERROR] No documents found. Please add files to the 'docs' folder.")
        exit(1)
    print(f"[INFO] Loaded {len(docs)} documents")

    print("[INFO] Splitting documents into chunks...")
    chunks = split_docs(docs)

    print("[INFO] Storing embeddings in Supabase...")
    vectorstore = build_vectorstore(chunks)

    print("[INFO] Setting up Gemini RAG pipeline...")
    qa = build_qa(vectorstore)

    print("\nChatbot ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = qa.invoke({"query": query})
        answer = response["result"]
        print("Bot:", answer)