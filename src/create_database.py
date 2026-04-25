"""
create_database.py — Ingests documents, chunks them, embeds them, and stores in ChromaDB
Inspired by: https://github.com/pixegami/langchain-rag-tutorial
Enhanced with: better metadata, progress tracking, and validation
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHROMA_DB_PATH, DOCUMENTS_PATH, CHUNK_SIZE,
    CHUNK_OVERLAP, EMBEDDING_MODEL, COLLECTION_NAME
)

console = Console()


def load_documents(documents_path: str) -> List[Document]:
    """Load all markdown and text documents from the given directory."""
    console.print(f"\n[bold cyan]📂 Loading documents from:[/bold cyan] {documents_path}")
    
    if not os.path.exists(documents_path):
        console.print(f"[red]ERROR: Documents path does not exist: {documents_path}[/red]")
        sys.exit(1)

    # Load markdown files
    md_loader = DirectoryLoader(
        documents_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = md_loader.load()

    # Also try .txt files
    try:
        txt_loader = DirectoryLoader(
            documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        documents += txt_loader.load()
    except Exception:
        pass

    console.print(f"[green]✓ Loaded {len(documents)} document(s)[/green]")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with overlap."""
    console.print(f"\n[bold cyan]✂️  Splitting documents...[/bold cyan]")
    console.print(f"   Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        # Extract a short source name
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["source_name"] = Path(src).stem

    console.print(f"[green]✓ Created {len(chunks)} chunks from {len(documents)} documents[/green]")
    return chunks


def get_embedding_function():
    """Return the HuggingFace embedding function (local, no API key needed)."""
    console.print(f"\n[bold cyan]🔢 Loading embedding model:[/bold cyan] {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    console.print("[green]✓ Embedding model loaded[/green]")
    return embeddings


def create_vector_store(chunks: List[Document], embeddings, reset: bool = False):
    """Create or update ChromaDB vector store."""
    if reset and os.path.exists(CHROMA_DB_PATH):
        console.print(f"\n[yellow]🗑️  Resetting existing database at {CHROMA_DB_PATH}[/yellow]")
        shutil.rmtree(CHROMA_DB_PATH)

    console.print(f"\n[bold cyan]💾 Creating vector database...[/bold cyan]")
    console.print(f"   Location: {CHROMA_DB_PATH}")
    console.print(f"   Collection: {COLLECTION_NAME}")
    console.print(f"   Documents to embed: {len(chunks)}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Embedding and storing chunks..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(chunks))

        # Process in batches to show progress
        batch_size = 50
        db = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if db is None:
                db = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_PATH,
                    collection_name=COLLECTION_NAME,
                )
            else:
                db.add_documents(batch)
            progress.advance(task, len(batch))

    return db


def print_database_summary(chunks: List[Document]):
    """Print a summary table of the database contents."""
    table = Table(title="📊 Database Summary", show_header=True, header_style="bold magenta")
    table.add_column("Source", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Avg Chunk Size", justify="right", style="yellow")

    from collections import defaultdict
    source_stats = defaultdict(lambda: {"count": 0, "total_size": 0})
    for chunk in chunks:
        src = chunk.metadata.get("source_name", "unknown")
        source_stats[src]["count"] += 1
        source_stats[src]["total_size"] += chunk.metadata.get("chunk_size", 0)

    total_chunks = 0
    for src, stats in sorted(source_stats.items()):
        avg_size = stats["total_size"] // stats["count"]
        table.add_row(src, str(stats["count"]), f"{avg_size} chars")
        total_chunks += stats["count"]

    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_chunks}[/bold]", "")
    console.print(table)


def main(reset: bool = True):
    console.print(Panel.fit(
        "[bold white]Hallucination-Aware RAG[/bold white]\n"
        "[dim]Database Creation Pipeline[/dim]",
        border_style="bright_blue"
    ))

    # 1. Load documents
    documents = load_documents(DOCUMENTS_PATH)
    if not documents:
        console.print("[red]No documents found! Add .md or .txt files to data/documents/[/red]")
        sys.exit(1)

    # 2. Split into chunks
    chunks = split_documents(documents)

    # 3. Load embedding model
    embeddings = get_embedding_function()

    # 4. Create vector store
    db = create_vector_store(chunks, embeddings, reset=reset)

    # 5. Summary
    print_database_summary(chunks)

    console.print(Panel(
        f"[green bold]✅ Database created successfully![/green bold]\n"
        f"[dim]Path: {CHROMA_DB_PATH}\n"
        f"Chunks stored: {len(chunks)}\n"
        f"Run query_rag.py to start querying[/dim]",
        border_style="green"
    ))

    return db


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create ChromaDB vector store")
    parser.add_argument("--no-reset", action="store_true", help="Don't reset existing DB")
    args = parser.parse_args()
    main(reset=not args.no_reset)