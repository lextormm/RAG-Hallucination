"""
evaluate.py — Evaluation Module for Hallucination-Aware RAG

Measures system performance on:
  - Hallucination Detection Accuracy
  - Self-Correction Effectiveness
  - Answer Quality (Faithfulness, Relevance)
  - Comparison: RAG vs. Hallucination-Aware RAG

Usage:
    python evaluate.py
    python evaluate.py --questions custom_questions.json
"""

import sys
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import HallucinationAwareRAG, RAGResponse
from config import OUTPUTS_PATH, HALLUCINATION_THRESHOLD

console = Console()


# ── Evaluation Test Cases ────────────────────────────────────────────────────

EVALUATION_QUESTIONS = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "category": "Definition",
        "expected_keywords": ["retrieval", "generation", "context", "documents", "LLM"],
    },
    {
        "question": "What are the main types of hallucinations in AI systems?",
        "category": "Hallucination Types",
        "expected_keywords": ["factual", "contextual", "intrinsic", "extrinsic"],
    },
    {
        "question": "What is supervised learning and how does it differ from unsupervised learning?",
        "category": "ML Fundamentals",
        "expected_keywords": ["labeled", "unlabeled", "training", "patterns"],
    },
    {
        "question": "What are the components of a RAG architecture?",
        "category": "RAG Architecture",
        "expected_keywords": ["vector", "embedding", "retriever", "generator", "chunker"],
    },
    {
        "question": "What is the consistency score in hallucination detection?",
        "category": "System-Specific",
        "expected_keywords": ["consistency", "score", "grounded", "context"],
    },
    {
        "question": "What were Leonardo da Vinci's contributions to quantum computing?",
        "category": "Trick Question",
        "expected_keywords": ["not", "no information", "context", "cannot"],
        "is_trick": True,  # Should trigger hallucination detection
    },
    {
        "question": "How does deep learning differ from traditional machine learning?",
        "category": "ML Comparison",
        "expected_keywords": ["neural", "layers", "deep", "features"],
    },
    {
        "question": "What are the limitations of current AI systems?",
        "category": "AI Limitations",
        "expected_keywords": ["data", "bias", "explainability", "brittle", "common sense"],
    },
]


@dataclass
class EvaluationResult:
    question: str
    category: str
    answer: str
    consistency_score: float
    confidence_level: str
    regeneration_count: int
    used_strict_mode: bool
    processing_time: float
    is_reliable: bool
    keyword_coverage: float  # How many expected keywords are in the answer
    is_trick: bool = False


class RAGEvaluator:
    """Evaluates the Hallucination-Aware RAG system across multiple dimensions."""

    def __init__(self, rag: HallucinationAwareRAG):
        self.rag = rag
        self.results: List[EvaluationResult] = []

    def run_evaluation(self, questions: List[Dict] = None) -> List[EvaluationResult]:
        """Run evaluation on all test questions."""
        questions = questions or EVALUATION_QUESTIONS
        
        console.print(Panel.fit(
            f"[bold white]RAG System Evaluation[/bold white]\n"
            f"[dim]Testing {len(questions)} questions across {len(set(q['category'] for q in questions))} categories[/dim]",
            border_style="bright_blue"
        ))

        self.results = []
        for i, q_data in enumerate(questions, 1):
            console.print(f"\n[bold magenta]Test {i}/{len(questions)}[/bold magenta] | Category: {q_data['category']}")
            console.print(f"[dim]Question: {q_data['question']}[/dim]")
            
            try:
                response = self.rag.query(q_data['question'], verbose=False)
                
                # Calculate keyword coverage
                answer_lower = response.answer.lower()
                expected_kw = q_data.get("expected_keywords", [])
                if expected_kw:
                    covered = sum(1 for kw in expected_kw if kw.lower() in answer_lower)
                    kw_coverage = covered / len(expected_kw)
                else:
                    kw_coverage = 1.0
                
                result = EvaluationResult(
                    question=q_data['question'],
                    category=q_data['category'],
                    answer=response.answer,
                    consistency_score=response.consistency_score,
                    confidence_level=response.confidence_label,
                    regeneration_count=response.regeneration_count,
                    used_strict_mode=response.used_strict_mode,
                    processing_time=response.processing_time_s,
                    is_reliable=response.is_reliable,
                    keyword_coverage=kw_coverage,
                    is_trick=q_data.get("is_trick", False),
                )
                
                self.results.append(result)
                
                # Quick status
                status = "✅" if result.is_reliable else "⚠️"
                console.print(
                    f"  {status} Score: {result.consistency_score:.2f} | "
                    f"Keywords: {kw_coverage*100:.0f}% | "
                    f"Regen: {result.regeneration_count} | "
                    f"Time: {result.processing_time:.1f}s"
                )
                
            except Exception as e:
                console.print(f"  [red]ERROR: {e}[/red]")

        return self.results

    def print_report(self):
        """Print a comprehensive evaluation report."""
        if not self.results:
            console.print("[red]No results to report[/red]")
            return

        console.print(Rule("[bold blue]Evaluation Report[/bold blue]"))

        # ── Per-question results table ───────────────────────────────────────
        table = Table(
            title="Question-by-Question Results",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        table.add_column("Category", style="cyan", max_width=20)
        table.add_column("Question", max_width=35)
        table.add_column("Score", justify="center")
        table.add_column("KW%", justify="center")
        table.add_column("Regen", justify="center")
        table.add_column("Time(s)", justify="center")
        table.add_column("Status", justify="center")

        for r in self.results:
            score_color = "green" if r.is_reliable else ("yellow" if r.consistency_score >= 0.4 else "red")
            q_short = r.question[:33] + "..." if len(r.question) > 35 else r.question
            table.add_row(
                r.category,
                q_short,
                f"[{score_color}]{r.consistency_score:.2f}[/{score_color}]",
                f"{r.keyword_coverage*100:.0f}%",
                str(r.regeneration_count),
                f"{r.processing_time:.1f}",
                "✅" if r.is_reliable else "⚠️"
            )
        
        console.print(table)

        # ── Aggregate metrics ────────────────────────────────────────────────
        total = len(self.results)
        reliable = sum(1 for r in self.results if r.is_reliable)
        avg_score = sum(r.consistency_score for r in self.results) / total
        avg_kw = sum(r.keyword_coverage for r in self.results) / total
        total_regens = sum(r.regeneration_count for r in self.results)
        strict_used = sum(1 for r in self.results if r.used_strict_mode)
        avg_time = sum(r.processing_time for r in self.results) / total
        
        # Trick question handling
        trick_results = [r for r in self.results if r.is_trick]
        trick_handled = sum(
            1 for r in trick_results
            if any(kw in r.answer.lower() for kw in ["not", "no information", "cannot", "don't"])
        )

        summary = Table(title="📊 Aggregate Metrics", box=box.ROUNDED)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="bold")
        summary.add_column("Notes")

        summary.add_row("Total Questions", str(total), "")
        summary.add_row(
            "Reliable Responses",
            f"[green]{reliable}/{total} ({reliable/total*100:.0f}%)[/green]",
            f"Score ≥ {HALLUCINATION_THRESHOLD}"
        )
        summary.add_row("Avg Consistency Score", f"{avg_score:.3f}", "Higher = more grounded")
        summary.add_row("Avg Keyword Coverage", f"{avg_kw*100:.1f}%", "Expected terms found")
        summary.add_row("Total Regenerations", str(total_regens), "Self-corrections performed")
        summary.add_row("Strict Mode Activations", str(strict_used), "Severe hallucination triggers")
        summary.add_row("Avg Processing Time", f"{avg_time:.2f}s", "Per query")
        if trick_results:
            summary.add_row(
                "Trick Questions Handled",
                f"{trick_handled}/{len(trick_results)}",
                "Correctly refused to hallucinate"
            )

        console.print(summary)

    def save_report(self) -> str:
        """Save the evaluation report to a JSON file."""
        os.makedirs(OUTPUTS_PATH, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUTS_PATH, f"evaluation_report_{timestamp}.json")
        
        data = {
            "timestamp": timestamp,
            "summary": {
                "total_questions": len(self.results),
                "reliable_responses": sum(1 for r in self.results if r.is_reliable),
                "avg_consistency_score": sum(r.consistency_score for r in self.results) / len(self.results),
                "avg_keyword_coverage": sum(r.keyword_coverage for r in self.results) / len(self.results),
                "total_regenerations": sum(r.regeneration_count for r in self.results),
            },
            "results": [
                {
                    "question": r.question,
                    "category": r.category,
                    "answer": r.answer,
                    "consistency_score": r.consistency_score,
                    "confidence_level": r.confidence_level,
                    "keyword_coverage": r.keyword_coverage,
                    "regeneration_count": r.regeneration_count,
                    "used_strict_mode": r.used_strict_mode,
                    "processing_time_s": r.processing_time,
                    "is_reliable": r.is_reliable,
                }
                for r in self.results
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[dim]📊 Evaluation report saved to: {filepath}[/dim]")
        return filepath


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the Hallucination-Aware RAG system")
    parser.add_argument("--questions", help="Path to custom questions JSON file")
    args = parser.parse_args()

    # Load custom questions if provided
    questions = None
    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)

    # Initialize and run
    rag = HallucinationAwareRAG()
    evaluator = RAGEvaluator(rag)
    evaluator.run_evaluation(questions)
    evaluator.print_report()
    evaluator.save_report()


if __name__ == "__main__":
    main()