# Multi-Agent Syllabus-Aware RAG System

A **multi-agent Retrieval-Augmented Generation (RAG)** pipeline built with **LangGraph** that answers student questions using **course syllabus material** while minimizing hallucinations through **multi-stage validation, critique, and feedback loops**.

The system combines **semantic retrieval, cross-encoder reranking, iterative answer refinement, and confidence evaluation** to produce syllabus appropriate responses.

Currently still **In-Progress**, trying to optimise run time and exploring other methods of measuring confidence.

---

# Overview

Traditional RAG systems often suffer from:

* Retrieving irrelevant documents
* Hallucinated answers when context is weak
* Overconfidence in generated responses

This project addresses these problems with a **multi-agent architecture** where specialized agents perform independent reasoning steps.

The system workflow:

1. **Query rewriting** – improves the student's question for retrieval.
2. **Retrieval decision** – determines whether syllabus retrieval is necessary. (To be edited)
3. **Vector retrieval** – fetches relevant syllabus chunks.
4. **Cross-encoder reranking** – ranks retrieved chunks by semantic relevance.
5. **Drafting agent** – generates an initial grounded answer.
6. **Critique agent** – evaluates answer quality and identifies weaknesses.
7. **Revision loop** – iteratively improves the answer.
8. **Confidence agent** – estimates final answer reliability. (To be edited)

This design significantly reduces hallucination risk and ensures responses remain grounded in course material.

---

# Architecture

The pipeline is implemented using **LangGraph**, which enables structured multi-agent workflows with conditional routing and revision loops.

---

# Key Features

### 1. Syllabus-Aware Retrieval

The system determines whether a question should use syllabus materials before performing retrieval, avoiding unnecessary vector searches for questions that can be answered from general knowledge.

---

### 2. Cross-Encoder Reranking

After retrieving documents, a **BGE CrossEncoder reranker** improves context quality by scoring each document relative to the query.

Model used:

* `BAAI/bge-reranker-large`

This step greatly improves answer grounding compared to pure vector similarity.

---

### 3. Iterative Answer Refinement

Instead of generating a single response, the system performs:

1. Initial answer draft
2. Professional critique
3. Revision cycles

---

### 4. Hallucination Reduction

Agents are instructed to:

* Use **only provided context**
* State when context is insufficient
* Remove unsupported claims during revision

---

### 5. Confidence Evaluation

A final agent estimates answer reliability using:

* Question
* Final answer
* Supporting context

It outputs a **confidence score from 0–10** representing the likelihood that the answer is correct and fully supported.

---

# Technologies Used

### Core Framework

* **LangGraph** – multi-agent workflow orchestration
* **LangChain** – LLM interface and tools

### Models

* **OpenAI GPT models** (`gpt-5-mini`) for reasoning agents
* **BGE CrossEncoder** for document reranking

### Retrieval

* Vector similarity search over syllabus

### Python Libraries

* see requirements.txt

---

# Pipeline behavior:

1. Query rewritten for clarity.
2. System determines retrieval is required.
3. Relevant syllabus material retrieved.
4. Cross-encoder reranks documents.
5. Draft answer generated.
6. Critique identifies missing explanations.
7. Revision improves clarity and completeness.
8. Final confidence score produced.

---

# Running the System

### 1. Install dependencies

```bash
pip install langchain langgraph sentence-transformers openai
```

---

### 2. Set OpenAI API key

```bash
export OPENAI_API_KEY="your_api_key"
```

or in Python:

```python
os.environ["OPENAI_API_KEY"] = "your_api_key"
```

---

### 3. Run the workflow

```python
result = graph.invoke(initial_state)
print(result["draft_answer"])
```

---

# Future Improvements

Potential extensions include:

* Allow educators to step in and help answer queries if agent is not confident in answering
* Multi-course routing
* Better syllabus topic classification
* Retrieval caching
* Token usage and latency monitoring

---

# Motivation

This project explores how **multi-agent reasoning pipelines** can improve the reliability of LLM-based tutoring systems by:

* Structuring reasoning steps
* Validating evidence
* Iteratively refining answers

The goal is to build **trustworthy AI assistants for education** that remain grounded in verified course material.

---
