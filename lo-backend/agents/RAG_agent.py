import os
import hashlib
from dotenv import load_dotenv
from supabase import create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or not GEMINI_API_KEY:
    print("[ERROR] One or more environment variables are missing!")
    exit(1)

# --- Valid roles ---
VALID_ROLES = ["student", "individual", "enterprise"]

# --- Docs folder (relative to this file) ---
DOCS_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "docs"))


# --- 1. Helpers ---
def make_chunk_id(text, source):
    raw = f"{source}:{text}"
    return hashlib.md5(raw.encode()).hexdigest()


def detect_role(filename):
    """Detect role from filename. Example: enterprise_level2_loops.pdf"""
    filename_lower = filename.lower()
    for role in VALID_ROLES:
        if role in filename_lower:
            return role
    return "student"  # default


# --- 2. Load documents ---
def load_docs():
    docs = []

    if not os.path.exists(DOCS_FOLDER):
        print(f"[ERROR] Docs folder '{DOCS_FOLDER}' not found!")
        return docs

    # Walk through role subfolders (enterprise/, student/, individual/)
    for role_folder in VALID_ROLES:
        folder_path = os.path.join(DOCS_FOLDER, role_folder)

        if not os.path.exists(folder_path):
            print(f"[WARN] Role folder not found, skipping: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)

            if file.endswith(".txt"):
                loaded_docs = TextLoader(path, encoding="utf-8").load()
            elif file.endswith(".pdf"):
                loaded_docs = PyPDFLoader(path).load()
            else:
                continue

            format_type = "pdf" if file.endswith(".pdf") else "txt"
            topic = file.replace(".pdf", "").replace(".txt", "").replace("_", " ")

            # Detect level from filename
            level = 1
            if "level2" in file:
                level = 2
            elif "level3" in file:
                level = 3

            difficulty = {1: "easy", 2: "medium", 3: "hard"}.get(level, "easy")

            # Role comes from the subfolder name, not just filename
            role = role_folder

            for doc in loaded_docs:
                doc.metadata.update({
                    "source": path,
                    "format": format_type,
                    "role": role,
                    "difficulty": difficulty,
                    "topic": topic,
                    "level": level,
                    "progress_level": 0
                })

            docs.extend(loaded_docs)
            print(f"[INFO] Loaded '{file}' — role='{role}', level={level}, difficulty='{difficulty}'")

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
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
        transport ="rest"
    )

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    vectorstore = SupabaseVectorStore(
        client=client,
        table_name="learning_chunks",
        embedding=embeddings,
        query_name="match_learning_chunks"
    )

    docs_to_store = []
    for chunk in chunks:
        metadata = chunk.metadata.copy()
        metadata["chunk_id"] = make_chunk_id(chunk.page_content, metadata.get("source", "unknown"))
        docs_to_store.append(Document(
            page_content=chunk.page_content,
            metadata=metadata
        ))

    vectorstore.add_documents(docs_to_store)
    print(f"[INFO] Stored {len(docs_to_store)} chunks in Supabase")
    return vectorstore


# --- 5. Build QA pipeline ---
def build_qa(vectorstore, role="student"):
    role_instructions = {
        "student": "Explain clearly and simply, as if teaching a student. Use examples and analogies.",
        "individual": "Be concise and practical. Focus on real-world application and self-learning.",
        "enterprise": "Be professional and thorough. Focus on scalability, best practices, and team use cases."
    }

    tone = role_instructions.get(role, role_instructions["student"])

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        You are an expert AI tutor.
        {tone}
        Use the following context to answer the question accurately and helpfully.
        You can merge the context with your own knowledge to provide relevant answers.

        Context:
        {{context}}

        Question:
        {{question}}

        Answer:
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
        print("[ERROR] No documents found. Please add files to the role subfolders.")
        exit(1)
    print(f"[INFO] Loaded {len(docs)} documents")

    print("[INFO] Splitting documents into chunks...")
    chunks = split_docs(docs)

    print("[INFO] Storing embeddings in Supabase...")
    vectorstore = build_vectorstore(chunks)

    print("[INFO] Setting up Gemini RAG pipeline...")
    qa = build_qa(vectorstore, role="student")

    print(f"\nChatbot ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = qa.invoke({"query": query})
        print("Bot:", response["result"])
