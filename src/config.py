"""
config.py — Central configuration for the Hallucination-Aware RAG system
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI API ──────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Paths ───────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documents")
OUTPUTS_PATH: str = "./outputs"

# ── Hallucination Detection ─────────────────────────────────────────────────
HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.6"))
STRICT_MODE_THRESHOLD: float = float(os.getenv("STRICT_MODE_THRESHOLD", "0.4"))
MAX_REGENERATION_ATTEMPTS: int = int(os.getenv("MAX_REGENERATION_ATTEMPTS", "3"))

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# ── Embedding model (local, no API key needed) ───────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── ChromaDB collection name ─────────────────────────────────────────────────
COLLECTION_NAME: str = "hallucination_aware_rag"

# ── Prompts ──────────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the context provided below to answer the user's question. If the context is missing details, do your best to provide a comprehensive answer by combining the context with your own knowledge.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

STRICT_RAG_PROMPT_TEMPLATE = """
You are an extremely precise and careful assistant. A previous answer was flagged for potential hallucination.
You MUST now generate a NEW answer following these STRICT rules:

STRICT RULES:
1. ONLY use information explicitly stated in the context below.
2. Every factual claim MUST be directly traceable to the context.
3. If a piece of information is not in the context, do NOT include it.
4. Do NOT infer, extrapolate, or assume anything beyond what is written.
5. If the context is insufficient, explicitly state: "The provided context does not contain enough information to fully answer this question."
6. Start each key fact with [FROM CONTEXT] to show grounding.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Strictly Grounded Answer:"""

HALLUCINATION_CHECK_PROMPT_TEMPLATE = """
You are a hallucination detection expert. Your task is to evaluate whether an AI-generated answer is consistent with the provided context.

--- CONTEXT ---
{context}
--- END CONTEXT ---

--- GENERATED ANSWER ---
{answer}
--- END ANSWER ---

Analyze the answer for hallucinations. A hallucination is any claim in the answer that:
1. Contradicts information in the context
2. Adds specific facts not present in the context
3. Misrepresents what the context says

INSTRUCTIONS:
1. Extract all distinct claims made in the generated answer.
2. For each claim, determine if it is 'supported' by the context or 'hallucinated'.
3. Calculate a partial consistency_score as the ratio: (number of supported claims) / (total number of claims). For example, if there are 3 supported claims and 1 hallucinated claim, the score should be 0.75. Do NOT simply output 0.0 or 1.0 unless it is 100% hallucinated or 100% supported.

Respond ONLY in the following JSON format (no other text):
{{
  "consistency_score": <float between 0.0 and 1.0 based on the ratio>,
  "has_hallucination": <true or false>,
  "hallucinated_claims": [<list of specific claims that are hallucinated>],
  "supported_claims": [<list of claims that ARE supported by context>],
  "reasoning": "<brief explanation including the claim counts and how the score was calculated>"
}}
"""