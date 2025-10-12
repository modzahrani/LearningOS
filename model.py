import os
import hashlib
from dotenv import load_dotenv
from supabase import create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.environ.get("supabase_url")
SUPABASE_KEY = os.environ.get("supabase_key")
GEMINI_API_KEY = os.environ.get("gemini_api_key")

if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
    print("[ERROR] One or more environment variables are missing!")
    exit(1)


# --- 1. Helpers ---
def make_chunk_id(text, source):
    raw = f"{source}:{text}"
    return hashlib.md5(raw.encode()).hexdigest()


# --- 2. Load documents ---
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


# --- 3. Split into chunks ---
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Created {len(chunks)} chunks")
    return chunks


# --- 4. Store embeddings in Supabase ---
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
        source = chunk.metadata.get("source", "unknown")

        metadata = {
            "source": source,
            "format": "pdf" if source.endswith(".pdf") else "txt",
            "role": "student",
            "difficulty": "easy",
            "topic": "general",
            "level": 1,
            "progress_level": 0
        }

        chunk_id = make_chunk_id(chunk.page_content, source)
        metadata["chunk_id"] = chunk_id

        doc = Document(
            page_content=chunk.page_content,
            metadata=metadata
        )
        docs_to_store.append(doc)

    vectorstore.add_documents(docs_to_store)
    print(f"[INFO] Stored {len(docs_to_store)} chunks in Supabase (deduplicated by chunk_id)")
    return vectorstore


# --- 5. Build QA pipeline ---
def build_qa(vectorstore):
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an expert AI tutor.
        Use the following context about the user to answer the question in a clear, concise, and helpful way.
        You can merge the context with your own knowledge to provide accurate and relevant answers.

        Context:
        {context}

        Question:
        {question}

        Answer in a friendly and informative tone:
        """
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa


# --- 6. Run chatbot ---
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
