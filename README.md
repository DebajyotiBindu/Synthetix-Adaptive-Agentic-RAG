# Synthetix- Adaptive Agentic Rag

Synthetix is an Agaptive Agentic RAG-based SaaS platform designed for autonomous medical information synthesis. It moves beyond simple "chat" by utilizing an intelligent orchestrator to manage complex workflows, search tool utilization, and context-aware retrieval.

*Note*-The RAG has been fed with medical data chunks so the Agent primarily can retrieve medical data and answer efficiently with respect to medical queries for now although with the use of web tool it can perform web search and provide answer on multiple contexts but still its highly advised to use this model for medical query purposes for now.I am trying to feed the RAG with more data,upgrade the dictionary used for storing frequent common questions to REDIS and change the instructions and guard rails provided to the LLM for increasing the model's usage across multiple domains but it will come in later updates.

## Medical Data (PDFs) used for this:
1. The New England Journal: Angiotensin–Neprilysin Inhibition versus Enalapril in Heart Failure
2. The New England journal of medicine: Pembrolizumab plus Axitinib versus Sunitinib for Advanced Renal-Cell Carcinoma
3. The New England journal of medicine: Dapagliflozin in Patients with Chronic Kidney Disease

## The Agentic Core
* Intelligent Orchestration: The FastAPI backend functions as an Agentic Orchestrator. It doesn't just pass text; it analyzes user intent, decides if it needs to search external web tools, and determines if it needs to retrieve data from the Vector Store.
* Web Tool Integration: The agents are empowered with autonomous web search capabilities to pull real-time data, bypassing the knowledge cutoff limits of standard LLMs.
* Adaptive RAG Pipeline: Instead of a static "retrieve and reply" flow, the system dynamically adjusts context based on the agent's internal assessment of what information is missing to provide a complete answer.
* Session-State Preservation: Django maintains the complex agentic state, ensuring that even if an agent takes three turns (Search -> Retrieve -> Synthesize) to answer a question, the user's conversation history remains coherent.

## How the Agentic Flow Works
* Request: User inputs a query via the Django frontend.
* Analysis: The FastAPI Agent analyzes if the query requires external knowledge (Web Tool) or proprietary internal data (RAG).
* Action: The Agent triggers the appropriate tool, manages the context window, and performs the synthesis.
* Response: The final, curated response is streamed back to the Django UI.

## Key Features
* Decoupled Micro-services: Independent deployment of Django and FastAPI ensures your web interface never blocks heavy AI processing.
* Session-Persistent RAG: Seamlessly merges historical conversation threads with real-time vector-retrieved context.
* Professional UI/UX: A high-tech, dark-mode interface with native markdown streaming for clean, readable medical/technical data visualization.
* Production-Ready: Built with Gunicorn, secure user-privilege management, and layered Docker builds.

## Tech Stack
* Backend: Python 3.11, Django, FastAPI
* AI/LLM: Custom RAG Pipeline
* Containerization: Docker & Docker Compose
* Web Server: Gunicorn (Production), Uvicorn (API)

## Getting Started
*Prerequisites*:
Docker Desktop (WSL2 required on Windows)

*Installation*:
Clone the repository:

```
git clone https://github.com/DebajyotiBindu/Synthetix-Adaptive-Agentic-RAG.git
cd Synthetix-Adaptive-Agentic-RAG
```

*Spin up the services*:

```
docker compose up --build
```

*Access the interfaces*:
* Web Interface (Django): http://localhost:8000
* API Engine (FastAPI): http://localhost:8001
