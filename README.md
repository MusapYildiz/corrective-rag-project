# 📘 Advanced RAG System with Hallucination Detection

This project implements an **advanced Retrieval-Augmented Generation (RAG)** pipeline using LangChain, Google Gemini (`gemini-1.5-flash`), Tavily Web Search, and custom hallucination & answer grading. It blends RAG with **structured LLM validation** to reduce hallucinations and ensure grounded answers.

---

## 🧠 System Architecture Overview

```
┌──────────────────────────────┐
│ User Query   │
└───│─────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│ Question     │
│ Routing      │◄────────────────────────────┐
└───│────────────────────┘             │
     │                       │
     ▼                       │
┌────────────────────────────┐      ┌────────────────────────┐
│ Web Search   │      │ Vector Search │
└───│────────────────────┘      └───│────────────────┘
     ▼                    ▼
┌───────────────────────────────────────────┐
│ Grade Retrieved Documents           │
└───│──────────────────────────┘
     ▼
┌────────────────────────────────────────────┐
│ Generate Answer (Google Gemini)     │
└───│──────────────────────────┘
     ▼
┌───────────────────────────────────────────┐
│ Hallucination Grader (LLM-based)    │
└───│──────────────────────────┘
     ▼
┌───────────────────────────────────────────┐
│ Answer Grader (LLM-based)           │
└───│────────────────────┐
     ▼             ▼
 "Useful"     "Not Useful" → Retry or Web Search
     │
     ▼
  ✅ Final Answer
```

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/MusapYildiz/corrective-rag-project.git
cd corrective-rag-project
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Add Environment Variables

Create a `.env` file with:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

---

```

---

## 🚀 Workflow: Step-by-Step

### Step 1: Create Vector Store from URLs

In `ingestion/vectorstore.py`, you load external documents and store them in a vector store using `FAISS` and Google embeddings.

### Step 2: Ask a Question

Run `main.py` and ask a natural question:

```bash
python main.py
```

### Step 3: Question Routing

The question is classified by the router LLM as:

* Web Search
* Vector Store Search

### Step 4: Document Retrieval

Based on the route:

* `retrieve.py`: uses Chroma
* `web_search.py`: uses Tavily API

### Step 5: Document Grading

System checks if retrieved documents are relevant. If not, a web search may be triggered.

### Step 6: Generate Answer

Uses Gemini (via LangChain) to generate an answer from the documents.

### Step 7: Hallucination Detection

Uses prompt-based verification to check if the answer is grounded in retrieved facts.

### Step 8: Answer Grading

Checks if the answer is actually useful and directly answers the original question.

---

## 🧪 Example Output

```bash
Hello Advanced RAG
---ROUTE QUESTION---
---ROUTE QUESTION TO WEB SEARCH---
---WEB SEARCH---
---GENERATE---
---CHECK HALLUCINATIONS---
---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
---GRADE GENERATION vs QUESTION---
---DECISION: GENERATION ADDRESSES QUESTION---
{'question': 'What is the weather in Istanbul today?', 'generation': 'The weather in Istanbul today is mostly sunny with a high of 28\u00b0C.'}
```

---

## ✅ Why This Project?

This project tackles a common issue in LLMs: **hallucinations**. By adding LLM-based grading for both factual grounding and relevance, this architecture improves trustworthiness and ensures better answer quality.

---

## 📎 Dependencies

* `LangChain`
* `langchain-google-genai`
* `Tavily`
* `LangGraph`
* `Pydantic`
* `dotenv`

---

## 📚 Future Improvements

* Add feedback loop to fine-tune routing decisions.
* Integrate user feedback for active learning.
* Use structured output parsers for detailed hallucination types.
