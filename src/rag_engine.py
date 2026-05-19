"""
rag_engine.py — The Hallucination-Aware RAG Engine

This is the heart of the system. It combines:
  1. Standard RAG (retrieve → generate)
  2. Hallucination Detection Layer (novel component)
  3. Feedback-Based Self-Correction Loop (novel component)

The self-correction loop:
  ┌─────────────────────────────────────────────────────┐
  │  Query → Retrieve Context → Generate Answer         │
  │              ↓                                      │
  │  Hallucination Detector checks consistency          │
  │              ↓                                      │
  │  Score ≥ threshold? → Return Answer [PASS]          │
  │  Score < threshold? → Regenerate with Constraints   │
  │              ↓ (repeat up to MAX_ATTEMPTS)          │
  │  Return best answer found + confidence score        │
  └─────────────────────────────────────────────────────┘
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import openai
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    TOP_K_RESULTS, HALLUCINATION_THRESHOLD,
    MAX_REGENERATION_ATTEMPTS,
    RAG_PROMPT_TEMPLATE, STRICT_RAG_PROMPT_TEMPLATE
)
from hallucination_detector import HallucinationDetector, HallucinationReport

console = Console()


@dataclass
class RAGResponse:
    """Complete response from the Hallucination-Aware RAG system."""
    
    query: str
    answer: str
    sources: List[str]
    consistency_score: float
    hallucination_report: HallucinationReport
    regeneration_count: int
    used_strict_mode: bool
    retrieved_chunks: List[Document] = field(default_factory=list)
    all_attempts: List[str] = field(default_factory=list)
    processing_time_s: float = 0.0
    
    @property
    def is_reliable(self) -> bool:
        return self.consistency_score >= HALLUCINATION_THRESHOLD
    
    @property
    def confidence_label(self) -> str:
        if self.consistency_score >= 0.85:
            return "HIGH"
        elif self.consistency_score >= HALLUCINATION_THRESHOLD:
            return "MEDIUM"
        elif self.consistency_score >= 0.4:
            return "LOW"
        else:
            return "VERY LOW"


class HallucinationAwareRAG:
    """
    Complete Hallucination-Aware RAG system with feedback-based self-correction.

    Usage:
        rag = HallucinationAwareRAG()
        response = rag.query("What is retrieval-augmented generation?")
        print(response.answer)
        print(f"Reliability: {response.consistency_score:.2f}")
    """

    def __init__(self):
        console.print("[dim]Initializing RAG engine...[/dim]")
        
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set!\n"
                "1. Copy .env.example to .env\n"
                "2. Add your key from https://platform.openai.com/api-keys"
            )
        
        # Initialize OpenAI
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = OPENAI_MODEL
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Load vector database
        self.db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )
        
        # Initialize hallucination detector
        self.detector = HallucinationDetector()
        
        # Ready to serve requests
        console.print("[green][SUCCESS] RAG engine ready[/green]")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, question: str, verbose: bool = True) -> RAGResponse:
        """
        Main entry point. Takes a user question, returns a RAGResponse
        with the answer, sources, and reliability metrics.
        """
        start_time = time.time()
        
        if verbose:
            console.print(Rule("[bold blue]Hallucination-Aware RAG Query[/bold blue]"))
            console.print(f"[bold]Question:[/bold] {question}\n")

        # Step 1: Retrieve relevant context
        chunks = self._retrieve(question, verbose=verbose)
        context_str = self._format_context(chunks)
        sources = self._extract_sources(chunks)

        # Step 2: Initial generation
        if verbose:
            console.print(f"\n[bold cyan][AI] Step 2: Generating Initial Answer...[/bold cyan]")
        
        answer = self._generate(question, context_str, strict=False)
        all_attempts = [answer]
        
        if verbose:
            console.print(Panel(answer, title="[bold]Initial Answer[/bold]", border_style="blue"))

        # Step 3: Hallucination detection + self-correction loop
        best_answer = answer
        best_score = 0.0
        best_report = None
        used_strict = False
        regen_count = 0

        for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 1):
            report = self.detector.detect(answer, context_str, attempt=attempt)
            
            # Track best answer
            if report.consistency_score > best_score:
                best_score = report.consistency_score
                best_answer = answer
                best_report = report

            # ✅ Answer is good enough — stop
            if not report.has_hallucination:
                if verbose:
                    console.print(f"\n[green bold][PASS] Answer accepted after {attempt} detection round(s)[/green bold]")
                break

            # [Retry] Hallucination detected — regenerate
            if attempt < MAX_REGENERATION_ATTEMPTS:
                regen_count += 1
                strict = report.needs_strict_mode
                used_strict = used_strict or strict
                
                if verbose:
                    mode = "STRICT" if strict else "STANDARD"
                    console.print(f"\n[yellow][Retry] Regenerating (attempt {attempt + 1}/{MAX_REGENERATION_ATTEMPTS}) — Mode: {mode}[/yellow]")
                
                answer = self._generate(
                    question, context_str,
                    strict=strict,
                    hallucinated_claims=report.hallucinated_claims
                )
                all_attempts.append(answer)
                
                if verbose:
                    console.print(Panel(
                        answer,
                        title=f"[bold]Regenerated Answer (attempt {attempt + 1})[/bold]",
                        border_style="yellow"
                    ))
            else:
                # Max attempts reached — use best answer found
                if verbose:
                    console.print(f"\n[yellow][WARN] Max attempts reached. Using best answer (score: {best_score:.2f})[/yellow]")

        elapsed = time.time() - start_time

        response = RAGResponse(
            query=question,
            answer=best_answer,
            sources=sources,
            consistency_score=best_score,
            hallucination_report=best_report,
            regeneration_count=regen_count,
            used_strict_mode=used_strict,
            retrieved_chunks=chunks,
            all_attempts=all_attempts,
            processing_time_s=elapsed,
        )

        if verbose:
            self._print_final_response(response)
        
        return response

    # ─────────────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, verbose: bool = True) -> List[Document]:
        """Retrieve top-K relevant chunks from ChromaDB."""
        if verbose:
            console.print(f"[bold cyan][Docs] Step 1: Retrieving Context (top {TOP_K_RESULTS} chunks)...[/bold cyan]")
        
        results = self.db.similarity_search_with_score(query, k=TOP_K_RESULTS)
        
        chunks = []
        if verbose:
            for i, (doc, score) in enumerate(results):
                src = doc.metadata.get("source_name", "unknown")
                console.print(f"   [{i+1}] {src} — similarity: {1 - score:.3f}")
                chunks.append(doc)
        else:
            chunks = [doc for doc, _ in results]
        
        return chunks

    def _format_context(self, chunks: List[Document]) -> str:
        """Format retrieved chunks into a context string."""
        parts = []
        for i, chunk in enumerate(chunks):
            src = chunk.metadata.get("source_name", "unknown")
            parts.append(f"[Source {i+1}: {src}]\n{chunk.page_content}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: List[Document]) -> List[str]:
        """Extract unique source names from chunks."""
        seen = set()
        sources = []
        for chunk in chunks:
            src = chunk.metadata.get("source_name", 
                  chunk.metadata.get("source", "unknown"))
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    def _generate(
        self,
        question: str,
        context: str,
        strict: bool = False,
        hallucinated_claims: Optional[List[str]] = None
    ) -> str:
        """Generate an answer using Gemini with the appropriate prompt template."""
        
        if strict:
            prompt = STRICT_RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )
            if hallucinated_claims:
                claims_str = "\n".join(f"  - {c}" for c in hallucinated_claims)
                prompt += f"\n\nWARNING: The following claims from the previous answer were flagged as HALLUCINATIONS. Do NOT include them:\n{claims_str}"
        else:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )

        for retry_attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2 if not strict else 0.05,
                    max_tokens=1024,
                )
                return response.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                if retry_attempt < 2:
                    console.print(f"[dim yellow]Rate limit reached. Waiting 5 seconds before retrying generation...[/dim yellow]")
                    time.sleep(5)
                    continue
                return f"[Error generating response: {e}]"
            except Exception as e:
                return f"[Error generating response: {e}]"

    def _print_final_response(self, response: RAGResponse):
        """Print the final formatted response."""
        color = response.hallucination_report.verdict_color if response.hallucination_report else "white"
        verdict = response.hallucination_report.verdict if response.hallucination_report else "UNKNOWN"
        
        console.print(Rule("[bold green]Final Response[/bold green]"))
        console.print(Panel(
            response.answer,
            title=f"[bold green]Final Answer[/bold green]",
            border_style="green"
        ))
        
        # Metadata panel
        meta_lines = [
            f"[{color}]Hallucination Check: {verdict}[/{color}]",
            f"Consistency Score:  [bold]{response.consistency_score:.2f}[/bold] / 1.00",
            f"Confidence Level:   [bold]{response.confidence_label}[/bold]",
            f"Regenerations:      {response.regeneration_count}",
            f"Strict Mode Used:   {'Yes' if response.used_strict_mode else 'No'}",
            f"Sources Used:       {', '.join(response.sources)}",
            f"Processing Time:    {response.processing_time_s:.2f}s",
        ]
        console.print(Panel("\n".join(meta_lines), title="[Info] Response Metadata", border_style="dim"))