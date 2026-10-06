# LearningOS

> An AI-powered adaptive learning platform that personalizes educational content, generates quizzes, and uses Retrieval-Augmented Generation (RAG) to provide context-aware learning experiences.

## Overview

**LearningOS** is an AI-powered learning platform designed to provide personalized and adaptive learning experiences.

The platform combines a modern **Next.js frontend**, **FastAPI backend**, **PostgreSQL/Supabase**, and **LLM-powered RAG pipelines** to help users learn based on their role, difficulty level, topic, and learning progress.

Instead of providing the same learning experience to every user, LearningOS uses user context and learning progress to dynamically tailor content and assessments.

### Key Goals

* Personalized learning paths
* AI-generated educational content
* Adaptive quizzes and assessments
* Context-aware AI responses using RAG
* Role-based learning experiences
* Progress-aware recommendations
* Secure authentication and data access

---

## Features

### 🤖 AI-Powered Learning

LearningOS uses LLMs to generate and process educational content based on the user's selected learning path and context.

### 📚 Retrieval-Augmented Generation

The platform uses a RAG pipeline to retrieve relevant educational content before generating responses.

The retrieval system uses:

* LangChain
* Hugging Face embeddings
* Supabase Vector Store
* PostgreSQL
* Metadata-based filtering

Documents are associated with metadata such as:

* Role
* Difficulty
* Format
* Topic
* Level
* Progress
* Source
* Content hash

This allows the system to retrieve content that is relevant to the user's current learning context rather than relying only on the LLM's general knowledge.

### 📝 Adaptive Quizzes

LearningOS generates quizzes based on the user's selected learning path and learning context.

Quiz information is persisted so users can continue their learning experience without losing their current state.

### 👤 Role-Based Learning

The platform supports different learning contexts, including:

* Student
* Individual
* Enterprise

Learning content can be filtered and adapted based on the selected role.

### 🔐 Authentication & Authorization

The backend provides authenticated API access using JWT-based authentication.

The application uses:

* HTTP-only cookie-based sessions
* JWT authentication
* Supabase authentication
* PostgreSQL Row Level Security (RLS)

### 📈 Learning Progress

User progress is incorporated into the learning experience, allowing content and assessments to be associated with the learner's current level and progress.

---

# Architecture

```text
┌─────────────────────────────┐
│          Next.js            │
│        Frontend             │
│                             │
│ React / TypeScript / UI     │
└──────────────┬──────────────┘
               │
               │ REST API
               ▼
┌─────────────────────────────┐
│          FastAPI            │
│          Backend            │
│                             │
│ Authentication              │
│ Business Logic              │
│ Learning APIs               │
│ Quiz Generation             │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌──────────────┐  ┌─────────────────┐
│ PostgreSQL   │  │   AI / RAG      │
│ + Supabase   │  │                 │
│              │  │ LangChain       │
│ User Data    │  │ Embeddings      │
│ Progress     │  │ Vector Search   │
│ Learning     │  │ LLM Generation  │
└──────────────┘  └─────────────────┘
```

---

# Tech Stack

## Frontend

* **Next.js**
* **React**
* **TypeScript**
* REST API integration
* Client-side learning state management

## Backend

* **Python**
* **FastAPI**
* **Uvicorn**
* **Pydantic**
* **asyncpg**
* **PostgreSQL**
* **Supabase**
* **JWT**
* **HTTP-only cookies**

## AI / Machine Learning

* **LangChain**
* **LangChain Google GenAI**
* **LangChain Hugging Face**
* **Hugging Face Transformers**
* **Sentence Transformers**
* **Supabase Vector Store**
* **Google Gemini**
* **PyTorch**
* **scikit-learn**

## Data & Infrastructure

* PostgreSQL
* Supabase
* Row Level Security (RLS)
* Vector search
* REST APIs

---

# RAG Pipeline

The RAG pipeline is one of the core components of LearningOS.

```text
Educational Content
       │
       ▼
Document Processing
       │
       ▼
Text Splitting
       │
       ▼
Hugging Face Embeddings
       │
       ▼
Supabase Vector Store
       │
       ▼
Metadata Filtering
       │
       ▼
Relevant Context
       │
       ▼
LLM
       │
       ▼
Personalized Response
```

The system uses metadata alongside vector similarity to improve retrieval relevance.

For example, a learner's request can be constrained by:

```text
Role       → Student
Difficulty → Intermediate
Topic      → Backend Development
Level      → 2
Progress   → 65%
```

This helps prevent unrelated educational material from being retrieved.

---

# Project Structure

```text
LearningOS/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── ...
│
├── backend/
│   ├── app/
│   ├── routes/
│   ├── services/
│   ├── models/
│   ├── utils/
│   └── ...
│
├── README.md
└── ...
```

> The exact structure may vary depending on the current branch/version of the project.

---

# Environment Variables

Create a `.env` file for the backend:

```env
DATABASE_URL=your_database_url

SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

GOOGLE_API_KEY=your_google_api_key

JWT_SECRET=your_jwt_secret
```

For the frontend, configure the required Next.js environment variables:

```env
NEXT_PUBLIC_API_URL=your_backend_url
```

**Never commit `.env` files or API keys to GitHub.**

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/modzahrani/LearningOS.git

cd LearningOS
```

---

## 2. Backend Setup

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

## 3. Frontend Setup

Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:3000
```

---

# Database

LearningOS uses **PostgreSQL through Supabase**.

The database is responsible for storing application data such as:

* Users
* Learning paths
* Learning progress
* Quiz information
* Educational content
* RAG metadata

Supabase Row Level Security is used to restrict database access according to the authenticated user and application rules.

---

# Authentication Flow

The application uses JWT-based authentication with cookie-based sessions.

```text
User
 │
 ▼
Login
 │
 ▼
Authentication
 │
 ▼
JWT
 │
 ▼
HTTP-only Cookie
 │
 ▼
Authenticated API Requests
 │
 ▼
FastAPI
 │
 ▼
Authorization / Database Access
```

This approach allows the frontend to communicate with the backend without exposing authentication tokens directly to client-side JavaScript.

---

# Learning Flow

A typical LearningOS user flow looks like this:

```text
User Registration
       │
       ▼
Select Learning Role
       │
       ▼
Select Learning Path
       │
       ▼
Determine Learning Context
       │
       ▼
Retrieve Relevant Content
       │
       ▼
AI-Generated Learning Material
       │
       ▼
Complete Quiz
       │
       ▼
Update Progress
       │
       ▼
Next Personalized Learning Step
```

---

# Client-Side State

LearningOS maintains selected learning and quiz state to allow the user to continue their current learning session.

Examples of persisted client-side state include:

```text
learningos_selected_path
learningos_quiz_id
```

---

# API

The backend exposes RESTful endpoints through FastAPI.

Example API interaction:

```text
Frontend
   │
   │ GET /api/...
   ▼
FastAPI
   │
   ├── Authentication
   ├── Business Logic
   ├── Database
   └── AI / RAG
   │
   ▼
JSON Response
```

FastAPI automatically provides interactive API documentation through:

```text
/docs
```

---

# Security

LearningOS implements several security mechanisms:

* JWT authentication
* HTTP-only cookies
* PostgreSQL Row Level Security
* Pydantic request validation
* Environment-based secret management
* User-specific data access

Sensitive credentials should always be stored in environment variables rather than committed to source control.

---

# Future Improvements

Potential future improvements include:

* More advanced adaptive learning algorithms
* Improved learner performance analytics
* More granular content recommendations
* Streaming AI responses
* Expanded enterprise learning functionality
* More advanced quiz difficulty adjustment
* Learning dashboards and analytics
* Improved evaluation of RAG retrieval quality
* Automated AI-generated learning plans

---

# What We Learned

Building LearningOS involved working across multiple areas of modern software engineering:

* Designing REST APIs with FastAPI
* Building full-stack applications with Next.js and Python
* PostgreSQL database design
* Supabase authentication and RLS
* JWT-based authentication
* Vector databases and semantic search
* Retrieval-Augmented Generation
* LLM integration
* Embedding models
* Metadata-based retrieval
* Async Python development
* Connecting AI systems with traditional application architecture

The project also provided practical experience integrating AI functionality into a production-style full-stack application rather than treating the LLM as an isolated feature.

---

# Authors

**Mohammed Alzahrani** **Khalaf Alshammrai** **Alaa Kenwai** **Thamer Aljohani** **Alawi Bafigeeh**

Software Engineering Graduate | Full-Stack Developer | AI

GitHub: [@modzahrani](https://github.com/modzahrani)

---

## License

This project is intended for educational and portfolio purposes.

