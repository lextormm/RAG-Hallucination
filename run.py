"""
run.py — One-Stop Entry Point for Hallucination-Aware RAG

Handles the full pipeline:
  1. Install dependencies (optional)
  2. Create vector database
  3. Run query / interactive / demo / evaluation mode

Usage:
    python run.py setup              # Create database
    python run.py query "Question?"  # Single query
    python run.py interactive        # Interactive chat
    python run.py demo               # Run demo questions
    python run.py evaluate           # Run evaluation suite
    python run.py test               # Run unit tests
"""

import sys
import os
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

# Add src to path
SRC_PATH = str(Path(__file__).parent / "src")
sys.path.insert(0, SRC_PATH)


def banner():
    console.print(Panel(
        "[bold white]🧠 Hallucination-Aware Retrieval-Augmented Generation[/bold white]\n"
        "[cyan]with Feedback-Based Self-Correction[/cyan]\n\n"
        "[dim]Novel system that detects and corrects LLM hallucinations\n"
        "using an iterative feedback loop powered by OpenAI API[/dim]",
        border_style="bright_blue",
        padding=(1, 4)
    ))


def check_env():
    """Check that .env file exists with API key."""
    env_file = Path(".env")
    if not env_file.exists():
        console.print("[yellow]⚠️  .env file not found. Copying from .env.example...[/yellow]")
        import shutil
        shutil.copy(".env.example", ".env")
        console.print("[red]❗ Please edit .env and add your OPENAI_API_KEY before continuing.[/red]")
        console.print("[dim]Get a key at: https://platform.openai.com/api-keys[/dim]")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "your_openai_api_key_here":
        console.print("[red]❗ OPENAI_API_KEY not set in .env file![/red]")
        return False
    
    console.print("[green]✓ API key found[/green]")
    return True


def cmd_setup():
    """Create the vector database."""
    console.print(Rule("[bold blue]Setup: Creating Vector Database[/bold blue]"))
    
    os.chdir(Path(__file__).parent)
    sys.path.insert(0, SRC_PATH)
    
    from create_database import main as create_db
    create_db(reset=True)


def cmd_query(question: str):
    """Run a single query."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from query_rag import run_single_query
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    run_single_query(rag, question)


def cmd_interactive():
    """Run interactive mode."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from query_rag import run_interactive
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    run_interactive(rag)


def cmd_demo():
    """Run demo questions."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from query_rag import run_demo
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    run_demo(rag)


def cmd_evaluate():
    """Run evaluation suite."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from evaluate import RAGEvaluator
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    evaluator = RAGEvaluator(rag)
    evaluator.run_evaluation()
    evaluator.print_report()
    evaluator.save_report()


def cmd_test():
    """Run unit tests."""
    console.print(Rule("[bold blue]Running Unit Tests[/bold blue]"))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=Path(__file__).parent
    )
    return result.returncode


def print_help():
    console.print("""
[bold]Usage:[/bold]
  python run.py [command] [options]

[bold]Commands:[/bold]
  [cyan]setup[/cyan]              Create the ChromaDB vector database from documents
  [cyan]query[/cyan] "Question"   Run a single query
  [cyan]interactive[/cyan]        Start interactive question-answering session
  [cyan]demo[/cyan]               Run built-in demo questions
  [cyan]evaluate[/cyan]           Run the full evaluation suite with metrics
  [cyan]test[/cyan]               Run unit tests

[bold]Examples:[/bold]
  python run.py setup
  python run.py query "What is hallucination in AI?"
  python run.py interactive
  python run.py demo
  python run.py evaluate

[bold]First-time setup:[/bold]
  1. cp .env.example .env
  2. Edit .env with your OpenAI API key
  3. python run.py setup
  4. python run.py demo
""")


def main():
    banner()
    
    if len(sys.argv) < 2:
        print_help()
        return
    
    cmd = sys.argv[1].lower()
    
    os.chdir(Path(__file__).parent)
    
    if cmd == "setup":
        cmd_setup()
    elif cmd == "query":
        if len(sys.argv) < 3:
            console.print("[red]Usage: python run.py query 'Your question here'[/red]")
        else:
            cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "interactive":
        cmd_interactive()
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "evaluate":
        cmd_evaluate()
    elif cmd == "test":
        sys.exit(cmd_test())
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        print_help()


if __name__ == "__main__":
    main()