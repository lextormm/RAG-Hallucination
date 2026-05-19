# Hallucination-Aware RAG System

This project implements a Retrieval-Augmented Generation (RAG) system augmented with a **Feedback-Based Self-Correction Loop**. This system not only retrieves context and generates answers but also actively evaluates its own generated answers for hallucinations and iteratively corrects them.

This documentation provides every technical detail required to understand the system and create a comprehensive system architecture diagram.

---
<img width="1012" height="532" alt="image" src="https://github.com/user-attachments/assets/d62fcb16-ce8c-42dd-91b1-13548dd8c96d" />

## 1. System Components & External API Calling

When mapping out the system diagram, the architecture consists of these primary boundaries:

### A. Local Storage & Processing (No external API calls)
1.  **Document Directory (`data/documents/`)**: Local storage for raw `.md` and `.txt` files.
2.  **ChromaDB Vector Store (`chroma_db/`)**: Local persistence directory storing chunks and embeddings.
3.  **Embedding Model (`sentence-transformers/all-MiniLM-L6-v2`)**: Runs completely locally on the CPU via HuggingFace to convert text chunks into vector embeddings. No API costs are incurred here.
4.  **Local Python Execution Engine**: Orchestrates the logic (LangChain chunking, loops, prompt formatting).

### B. External API Boundaries (OpenAI API)
The system communicates with the external OpenAI API (`gpt-4o-mini` by default) at two distinct points in the workflow:
1.  **Generation API Call (`rag_engine._generate()`)**:
    *   **Input**: System prompt (Standard or Strict) + Retrieved Context String + User Question.
    *   **Output**: Generated Text Answer.
2.  **Hallucination Detection API Call (`hallucination_detector.detect()`)**:
    *   **Input**: Detection prompt + Retrieved Context String + Generated Answer from step 1.
    *   **Output**: JSON object containing `consistency_score`, boolean flags, hallucinated claims, and reasoning.

---

## 2. Detailed Data Flow & Execution Pipeline

To draw the system diagram, follow this exact step-by-step data flow:

### Phase 1: Data Ingestion & Setup (`create_database.py`)
1.  **Input**: Raw `.md` and `.txt` documents from `data/documents/`.
2.  **Process**: `DirectoryLoader` reads files -> `RecursiveCharacterTextSplitter` chunks text (Size: 800, Overlap: 100) and extracts metadata (source names).
3.  **Embedding**: Chunks are passed to `HuggingFaceEmbeddings` model (`all-MiniLM-L6-v2`), generating vector representations locally.
4.  **Storage**: Vectors and metadata are saved to the local ChromaDB database.

### Phase 2: Query Execution & Self-Correction Loop (`rag_engine.py`)
1.  **Query Input**: User submits a question string.
2.  **Retrieval**:
    *   The question is embedded locally via HuggingFace.
    *   The system performs a `similarity_search_with_score` against ChromaDB.
    *   Top-K (default: 5) context chunks are retrieved.
3.  **Initial Generation (Attempt 1)**:
    *   **Inputs**: User Question + Top-K Contexts.
    *   **Action**: Calls OpenAI API using `RAG_PROMPT_TEMPLATE`.
    *   **Output**: Initial Answer String.
4.  **Hallucination Detection (`hallucination_detector.py`)**:
    *   **Inputs**: Initial Answer String + Top-K Contexts.
    *   **Action**: Calls OpenAI API with `HALLUCINATION_CHECK_PROMPT_TEMPLATE` enforcing JSON output.
    *   **Output**: `HallucinationReport` (Consistency Score 0.0 - 1.0, lists of supported/hallucinated claims).
5.  **Evaluation & Self-Correction Routing**:
    *   **Condition A (Pass)**: If `consistency_score >= 0.6` (`HALLUCINATION_THRESHOLD`). The loop breaks, and the answer is returned to the user.
    *   **Condition B (Fail/Regenerate)**: If `consistency_score < 0.6`. The engine triggers the loop.
6.  **Regeneration (Attempt N)**:
    *   **Inputs**: User Question + Top-K Contexts + Flagged Hallucinated Claims.
    *   **Action**: Calls OpenAI API using `STRICT_RAG_PROMPT_TEMPLATE` (which explicitly forbids the flagged claims).
    *   **Loop**: Output goes back to Step 4. Repeats up to `MAX_REGENERATION_ATTEMPTS` (default: 3).
7.  **Final Output**: The best answer found across all attempts is returned along with confidence metadata and sources.

---

## 3. Directory Structure & File Roles

*   **`run.py`**: The main entry point CLI (Setup, Query, Demo, Evaluate).
*   **`src/config.py`**: Configuration constants, OpenAI model strings, thresholds, and all 3 prompt templates.
*   **`src/create_database.py`**: LangChain ingestion, chunking, and ChromaDB instantiation.
*   **`src/rag_engine.py`**: The `HallucinationAwareRAG` class containing the generation and self-correction loop logic.
*   **`src/hallucination_detector.py`**: The `HallucinationDetector` class that parses the JSON output from the LLM evaluator.
*   **`src/query_rag.py`**: Handles individual/interactive queries and saves JSON output reports of the queries.
*   **`src/evaluate.py`**: Evaluation suite running predefined test/trick questions to measure keyword coverage, regeneration counts, and consistency.
*   **`tests/`**: Unit tests (e.g., `test_hallucination_detector.py` mocks LangChain/OpenAI to test detector logic offline).

---

## 4. Key Metrics and Thresholds (from `config.py`)

*   `CHUNK_SIZE`: 800
*   `CHUNK_OVERLAP`: 100
*   `TOP_K_RESULTS`: 5
*   `HALLUCINATION_THRESHOLD`: 0.6 (Minimum score to be considered 'reliable')
*   `STRICT_MODE_THRESHOLD`: 0.4 (Score below which the prompt switches to strict mode)
*   `MAX_REGENERATION_ATTEMPTS`: 3

## 5. Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` from `.env.example` and add `OPENAI_API_KEY`.
3. Add documents to `data/documents/`.
4. Initialize database: `python run.py setup`
5. Run a query: `python run.py query "Your question here"`
6. Run interactive mode: `python run.py interactive`
