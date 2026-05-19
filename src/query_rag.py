"""
query_rag.py — Command-line interface for the Hallucination-Aware RAG system

Usage:
    python query_rag.py "What is RAG and how does it reduce hallucinations?"
    python query_rag.py --interactive
    python query_rag.py --demo
"""

import sys
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import HallucinationAwareRAG, RAGResponse
from config import OUTPUTS_PATH

console = Console()

# Pre-built demo questions to showcase the system
DEMO_QUESTIONS = [
    "What is Retrieval-Augmented Generation and what are its main components?",
    "How does hallucination detection work in AI systems?",
    "What are the different types of machine learning?",
    "What are the benefits of RAG over pure LLMs?",
    "What is the hallucination problem in large language models?",
]


def save_response(response: RAGResponse, output_dir: str = OUTPUTS_PATH):
    """Save a RAG response to a JSON file for analysis."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    data = {
        "timestamp": timestamp,
        "query": response.query,
        "answer": response.answer,
        "sources": response.sources,
        "consistency_score": response.consistency_score,
        "confidence_level": response.confidence_label,
        "is_reliable": response.is_reliable,
        "regeneration_count": response.regeneration_count,
        "used_strict_mode": response.used_strict_mode,
        "processing_time_s": response.processing_time_s,
        "hallucination_report": {
            "verdict": response.hallucination_report.verdict,
            "has_hallucination": response.hallucination_report.has_hallucination,
            "hallucinated_claims": response.hallucination_report.hallucinated_claims,
            "supported_claims": response.hallucination_report.supported_claims,
            "reasoning": response.hallucination_report.reasoning,
        } if response.hallucination_report else None,
        "all_attempts": response.all_attempts,
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def run_single_query(rag: HallucinationAwareRAG, question: str, save: bool = True) -> RAGResponse:
    """Run a single query and optionally save the output."""
    response = rag.query(question, verbose=True)
    
    if save:
        filepath = save_response(response)
        console.print(f"\n[dim][Saved] Response saved to: {filepath}[/dim]")
    
    return response


def run_interactive(rag: HallucinationAwareRAG):
    """Run an interactive question-answering session."""
    console.print(Panel.fit(
        "[bold white]Interactive Mode[/bold white]\n"
        "[dim]Type your questions below. Enter 'quit' to exit.[/dim]",
        border_style="bright_blue"
    ))
    
    session_responses = []
    
    while True:
        try:
            question = Prompt.ask("\n[bold cyan]Your Question[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if question.lower() in ("quit", "exit", "q"):
            break
        
        if not question:
            continue
        
        response = run_single_query(rag, question, save=True)
        session_responses.append(response)
    
    # Session summary
    if session_responses:
        _print_session_summary(session_responses)


def run_demo(rag: HallucinationAwareRAG):
    """Run all demo questions to showcase the system."""
    # Print welcome message for the demo mode
    console.print(Panel.fit(
        "[bold white]Demo Mode[/bold white]\n"
        f"[dim]Running {len(DEMO_QUESTIONS)} pre-built demo questions[/dim]",
        border_style="bright_blue"
    ))
    
    all_responses = []
    for i, question in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"\n[bold magenta]Demo {i}/{len(DEMO_QUESTIONS)}[/bold magenta]")
        response = run_single_query(rag, question, save=True)
        all_responses.append(response)
        console.print("\n" + "─" * 80)
    
    _print_session_summary(all_responses)
    
    # Save summary report
    _save_demo_report(all_responses)


def _print_session_summary(responses: list):
    """Print a summary table of all responses in a session."""
    console.print(Rule("[bold blue]Session Summary[/bold blue]"))
    
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Query (truncated)", style="cyan", max_width=40)
    table.add_column("Score", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Regen", justify="center")
    table.add_column("Status", justify="center")
    
    for r in responses:
        score_color = "green" if r.is_reliable else "red"
        table.add_row(
            r.query[:38] + "..." if len(r.query) > 40 else r.query,
            f"[{score_color}]{r.consistency_score:.2f}[/{score_color}]",
            r.confidence_label,
            str(r.regeneration_count),
            "[PASS]" if r.is_reliable else "[WARN]"
        )
    
    console.print(table)
    
    avg_score = sum(r.consistency_score for r in responses) / len(responses)
    reliable_count = sum(1 for r in responses if r.is_reliable)
    console.print(f"\n[bold]Average Consistency: {avg_score:.2f} | Reliable Responses: {reliable_count}/{len(responses)}[/bold]")


def _save_demo_report(responses: list):
    """Save a comprehensive demo report."""
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUTS_PATH, f"demo_report_{timestamp}.json")
    
    report = {
        "timestamp": timestamp,
        "total_queries": len(responses),
        "avg_consistency_score": sum(r.consistency_score for r in responses) / len(responses),
        "reliable_responses": sum(1 for r in responses if r.is_reliable),
        "total_regenerations": sum(r.regeneration_count for r in responses),
        "results": [
            {
                "query": r.query,
                "answer": r.answer,
                "consistency_score": r.consistency_score,
                "confidence": r.confidence_label,
                "regeneration_count": r.regeneration_count,
                "sources": r.sources,
            }
            for r in responses
        ]
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[dim][Saved] Demo report saved to: {filepath}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Hallucination-Aware RAG — Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_rag.py "What is RAG?"
  python query_rag.py --interactive
  python query_rag.py --demo
        """
    )
    parser.add_argument("question", nargs="?", help="Question to ask the RAG system")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--demo", "-d", action="store_true", help="Run demo questions")
    parser.add_argument("--no-save", action="store_true", help="Don't save response to file")
    
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold white][System] Hallucination-Aware RAG[/bold white]\n"
        "[dim]with Feedback-Based Self-Correction[/dim]",
        border_style="bright_blue"
    ))

    # Initialize RAG engine
    try:
        rag = HallucinationAwareRAG()
    except ValueError as e:
        console.print(f"[red bold]Configuration Error:[/red bold]\n{e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red bold]Initialization Error:[/red bold]\n{e}")
        console.print("[yellow]Make sure you've run create_database.py first![/yellow]")
        sys.exit(1)

    # Dispatch to appropriate mode
    if args.demo:
        run_demo(rag)
    elif args.interactive:
        run_interactive(rag)
    elif args.question:
        run_single_query(rag, args.question, save=not args.no_save)
    else:
        parser.print_help()
        console.print("\n[yellow]Tip: Try --demo to see the system in action![/yellow]")


if __name__ == "__main__":
    main()