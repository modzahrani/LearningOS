import os
import re
import hashlib
from functools import lru_cache
from typing import Any
from dotenv import load_dotenv
from supabase import create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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

# --- Config ---
VALID_ROLES = ["student", "individual", "enterprise"]
EMBED_BATCH_SIZE = 100  # chunks per embedding batch — tune up/down as needed

DOCS_FOLDER = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
)


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------

def make_chunk_id(text: str, source: str) -> str:
    raw = f"{source}:{text}"
    return hashlib.md5(raw.encode()).hexdigest()


def _require_env() -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or not GEMINI_API_KEY:
        raise RuntimeError(
            "SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and GEMINI_API_KEY must be set"
        )


def parse_level(filename: str) -> int:
    """
    FIX: extract level number with regex so level8, level9, level10, etc.
    are handled correctly instead of all falling back to 1.
    """
    match = re.search(r"level(\d+)", filename.lower())
    if match:
        return int(match.group(1))
    return 1  # default


def level_to_difficulty(level: int) -> str:
    if level <= 2:
        return "easy"
    elif level <= 5:
        return "medium"
    else:
        return "hard"


# ---------------------------------------------------------------------------
# 2. Load documents
# ---------------------------------------------------------------------------

def load_docs() -> list[Document]:
    docs = []

    if not os.path.exists(DOCS_FOLDER):
        print(f"[ERROR] Docs folder '{DOCS_FOLDER}' not found!")
        return docs

    for role_folder in VALID_ROLES:
        folder_path = os.path.join(DOCS_FOLDER, role_folder)

        if not os.path.exists(folder_path):
            print(f"[WARN] Role folder not found, skipping: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)

            if file.endswith(".txt"):
                loaded_docs = TextLoader(path, encoding="utf-8").load()
                format_type = "txt"
            elif file.endswith(".pdf"):
                loaded_docs = PyPDFLoader(path).load()
                format_type = "pdf"
            else:
                continue

            # FIX: use regex-based level parsing
            level = parse_level(file)
            difficulty = level_to_difficulty(level)
            topic = re.sub(r"\.(pdf|txt)$", "", file).replace("_", " ")

            for doc in loaded_docs:
                doc.metadata.update({
                    "source": path,
                    "format": format_type,
                    "role": role_folder,
                    "difficulty": difficulty,
                    "topic": topic,
                    "level": level,
                    "progress_level": 0,
                })

            docs.extend(loaded_docs)
            print(
                f"[INFO] Loaded '{file}' — "
                f"role='{role_folder}', level={level}, difficulty='{difficulty}'"
            )

    return docs


# ---------------------------------------------------------------------------
# 3. Split into chunks
# ---------------------------------------------------------------------------

def split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # FIX: larger chunks → fewer total chunks → faster ingestion
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Created {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# 4. Fetch already-indexed chunk IDs from Supabase
# ---------------------------------------------------------------------------

def fetch_existing_chunk_ids(client) -> set[str]:
    """
    FIX: pull every chunk_id already stored so we can skip re-embedding them.
    Uses pagination to handle large tables safely.
    """
    existing = set()
    page_size = 1000
    offset = 0

    while True:
        response = (
            client.table("learning_chunks")
            .select("metadata->chunk_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            cid = row.get("chunk_id")
            if cid:
                existing.add(cid)

        if len(rows) < page_size:
            break
        offset += page_size

    print(f"[INFO] Found {len(existing)} already-indexed chunk IDs in Supabase")
    return existing


# ---------------------------------------------------------------------------
# 5. Store embeddings in Supabase — batched + duplicate-aware
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: list[Document]) -> SupabaseVectorStore:
    _require_env()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
        transport="rest",
    )

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    vectorstore = SupabaseVectorStore(
        client=client,
        table_name="learning_chunks",
        embedding=embeddings,
        query_name="match_learning_chunks",
    )

    # FIX: skip chunks that are already in the database
    existing_ids = fetch_existing_chunk_ids(client)

    docs_to_store = []
    skipped = 0
    for chunk in chunks:
        metadata = chunk.metadata.copy()
        chunk_id = make_chunk_id(chunk.page_content, metadata.get("source", "unknown"))
        metadata["chunk_id"] = chunk_id

        if chunk_id in existing_ids:
            skipped += 1
            continue

        docs_to_store.append(Document(page_content=chunk.page_content, metadata=metadata))

    print(f"[INFO] Skipping {skipped} already-indexed chunks")
    print(f"[INFO] Embedding {len(docs_to_store)} new chunks...")

    if not docs_to_store:
        print("[INFO] Nothing new to embed. Vectorstore is up to date.")
        return vectorstore

    # FIX: embed in small batches to avoid timeouts and memory pressure
    total = len(docs_to_store)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = docs_to_store[i : i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        print(f"[INFO] Batch {batch_num}/{total_batches} — embedding {len(batch)} chunks...")
        vectorstore.add_documents(batch)

    print(f"[INFO] Done. Stored {len(docs_to_store)} new chunks in Supabase.")
    return vectorstore


@lru_cache(maxsize=1)
def get_vectorstore() -> SupabaseVectorStore:
    """Return the existing Supabase-backed vector store without re-ingesting docs."""
    _require_env()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
        transport="rest",
    )
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return SupabaseVectorStore(
        client=client,
        table_name="learning_chunks",
        embedding=embeddings,
        query_name="match_learning_chunks",
    )


# ---------------------------------------------------------------------------
# 6. Build QA pipeline
# ---------------------------------------------------------------------------

def build_qa(vectorstore: SupabaseVectorStore, role: str = "student") -> RetrievalQA:
    role_instructions = {
        "student": "Explain clearly and simply, as if teaching a student. Use examples and analogies.",
        "individual": "Be concise and practical. Focus on real-world application and self-learning.",
        "enterprise": "Be professional and thorough. Focus on scalability, best practices, and team use cases.",
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
""",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
    )
    return qa


@lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    _require_env()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY,
    )


def _retrieval_prompt(role: str, level: int | None = None) -> PromptTemplate:
    role_instructions = {
        "student": "Explain clearly and simply, as if teaching a student. Use examples and analogies.",
        "individual": "Be concise and practical. Focus on real-world application and self-learning.",
        "enterprise": "Be professional and thorough. Focus on scalability, best practices, and team use cases.",
    }
    tone = role_instructions.get(role, role_instructions["student"])
    level_note = (
        f"The learner is currently around level {level}. Adjust depth accordingly."
        if level is not None
        else ""
    )

    return PromptTemplate(
        input_variables=["lesson_context", "context", "question"],
        template=f"""
You are LearningOS's AI tutor.
{tone}
{level_note}
Answer using the CURRENT LESSON CONTEXT first when it is relevant to the question.
Then use the retrieved lesson context.
Only fill gaps carefully with general AI knowledge when needed.
If the context is thin, be honest about that.
Respond in plain text only.
Do not use markdown.
Do not use asterisks, bold markers, or headings.
Keep the answer clear, direct, and educational.
Keep the answer concise.
Use at most 4 short sentences unless the user explicitly asks for more detail.
If the current lesson context is provided, prefer it over general knowledge.
If the answer is not supported by the current lesson context, say that clearly.

Current lesson context:
{{lesson_context}}

Context:
{{context}}

Question:
{{question}}

Answer:
""",
    )

def _clean_plain_text_answer(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""

    text = text.replace("***", "").replace("**", "").replace("__", "")
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def answer_question(
    query: str,
    role: str = "student",
    level: int | None = None,
    current_lesson_context: str | None = None,
    current_lesson_title: str | None = None,
    current_lesson_source: str | None = None,
    strict_lesson_focus: bool = False,
) -> dict[str, Any]:
    """Answer a learner question using the existing RAG corpus."""
    _require_env()
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty")

    vectorstore = get_vectorstore()

    docs: list[Document] = []
    if not strict_lesson_focus:
        try:
            docs = vectorstore.similarity_search(cleaned_query, k=8)
        except Exception:
            docs = []

    prioritized_docs: list[Document] = []
    remaining_docs: list[Document] = []
    seen_keys: set[str] = set()

    for doc in docs:
        source = str(doc.metadata.get("source") or "")
        role_match = str(doc.metadata.get("role") or role) == role
        key = f"{source}:{doc.metadata.get('chunk_id') or doc.page_content[:40]}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if current_lesson_source and source == current_lesson_source:
          prioritized_docs.append(doc)
        elif strict_lesson_focus:
          continue
        elif role_match:
          remaining_docs.append(doc)
        else:
          remaining_docs.append(doc)

    selected_docs = prioritized_docs[:2]
    for doc in remaining_docs:
        if len(selected_docs) >= 4:
            break
        if doc not in selected_docs:
            selected_docs.append(doc)

    if not selected_docs and docs:
        selected_docs = docs[:4]

    context = "\n\n".join(
        f"Source: {doc.metadata.get('topic', 'Untitled lesson')}\n{doc.page_content}"
        for doc in selected_docs
    )
    lesson_context = (current_lesson_context or "").strip()
    if current_lesson_title and lesson_context:
        lesson_context = f"Lesson: {current_lesson_title}\n{lesson_context}"

    if strict_lesson_focus and lesson_context:
        context = ""

    prompt = _retrieval_prompt(role, level).format(
        lesson_context=lesson_context or "No current lesson context was provided.",
        context=context or "No additional lesson context was retrieved.",
        question=cleaned_query,
    )
    response = _get_llm().invoke(prompt)
    raw_answer = response.content if hasattr(response, "content") else str(response)
    answer = _clean_plain_text_answer(str(raw_answer))

    sources = []
    seen = set()
    if current_lesson_source and current_lesson_title:
        seen.add(current_lesson_source)
        sources.append(
            {
                "source": current_lesson_source,
                "topic": current_lesson_title,
                "role": role,
                "level": int(level or 1),
            }
        )

    for doc in selected_docs:
        source = str(doc.metadata.get("source") or "")
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(
            {
                "source": source,
                "topic": str(doc.metadata.get("topic") or "Untitled lesson"),
                "role": str(doc.metadata.get("role") or role),
                "level": int(doc.metadata.get("level") or 1),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# 7. Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _require_env()
    print("[INFO] Loading documents...")
    docs = load_docs()
    if not docs:
        print("[ERROR] No documents found. Please add files to the role subfolders.")
        exit(1)
    print(f"[INFO] Loaded {len(docs)} document pages")

    print("[INFO] Splitting documents into chunks...")
    chunks = split_docs(docs)

    print("[INFO] Storing embeddings in Supabase (batched, duplicate-safe)...")
    vectorstore = build_vectorstore(chunks)

    print("[INFO] Setting up Gemini RAG pipeline...")
    qa = build_qa(vectorstore, role="student")

    print("\nChatbot ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = qa.invoke({"query": query})
        print("Bot:", response["result"])
