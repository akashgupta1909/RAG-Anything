"""rag-search-engine command-line interface.

uv run python main.py ingest                 # fetch, chunk, embed and index all domains
uv run python main.py ask "How do I ..."     # agentic RAG answer with citations
uv run python main.py search "query"         # retrieval only (no LLM)
uv run python main.py diagrams "load balancer"   # multimodal diagram search
uv run python main.py eval                   # retrieval metrics on the golden set
uv run python main.py serve                  # start the FastAPI server
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TextColumn
from rich.table import Table

from rag.config import get_settings
from rag.container import Container
from rag.utils import setup_logging

app = typer.Typer(
    help="Agentic RAG over technical documentation.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()

DomainOpt = Annotated[
    list[str] | None,
    typer.Option("--domain", "-d", help="Restrict to a domain (repeatable)."),
]


def _container() -> Container:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_dir)
    return Container(settings)


@app.command()
def ingest(
    domain: DomainOpt = None,
    refresh: Annotated[
        bool, typer.Option(help="Re-download / pull the sources.")
    ] = False,
    recreate: Annotated[
        bool, typer.Option(help="Drop and rebuild the collections.")
    ] = False,
) -> None:
    """Fetch, chunk, embed and index documents (incremental: unchanged docs are skipped)."""
    container = _container()
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        tasks: dict[str, TaskID] = {}

        def on_progress(stage: str, done: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = progress.add_task(f"Indexing {stage}", total=total)
            progress.update(tasks[stage], completed=done)

        report = container.ingestion.run(
            domain, refresh_sources=refresh, recreate=recreate, on_progress=on_progress
        )
    table = Table(title="Ingestion report", show_header=False)
    for key, value in vars(report).items():
        table.add_row(key.replace("_", " "), str(value))
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    domain: DomainOpt = None,
    stream: Annotated[
        bool, typer.Option(help="Stream the answer token by token.")
    ] = True,
    trace: Annotated[bool, typer.Option(help="Show the pipeline trace.")] = False,
) -> None:
    """Answer a question with the full agentic RAG pipeline."""
    pipeline = _container().pipeline
    if stream:
        done: dict[str, object] = {}
        sources: list[dict[str, object]] = []
        diagrams: list[dict[str, object]] = []
        for event in pipeline.stream(question, domains=domain):
            match event["type"]:
                case "plan":
                    plan = event["plan"]
                    console.print(
                        f"[dim]domains: {', '.join(plan['domains'])} | "
                        f"queries: {plan['sub_queries']} | rounds: {event['rounds']}[/dim]\n"
                    )
                case "token":
                    console.print(event["text"], end="", markup=False, highlight=False)
                case "sources":
                    sources = event["citations"]
                case "diagrams":
                    diagrams = event["diagrams"]
                case "done":
                    done = event
        console.print("\n")
        cited = set(done.get("cited", []))  # type: ignore[arg-type]
        _print_sources([s for s in sources if s["id"] in cited] or sources)
        _print_diagrams(diagrams)
        if trace:
            console.print_json(data=done.get("trace"))
        return

    response = pipeline.answer(question, domains=domain)
    console.print(Markdown(response.answer))
    _print_sources([c.model_dump() for c in response.citations if c.cited])
    _print_diagrams([d.model_dump() for d in response.diagrams])
    if trace:
        console.print_json(data=response.trace.model_dump())


def _print_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        return
    console.print("[bold]Sources[/bold]")
    for s in sources:
        heading = f" > {s['heading']}" if s.get("heading") else ""
        console.print(
            f"  [{s['id']}] {s['title']}{heading}\n      [link]{s['url']}[/link]"
        )


def _print_diagrams(diagrams: list[dict[str, object]]) -> None:
    if diagrams:
        console.print("[bold]Diagrams[/bold]")
        for d in diagrams:
            console.print(f"  - {d['caption']}: [link]{d['url']}[/link]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    domain: DomainOpt = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1)] = 5,
    mode: Annotated[Literal["hybrid", "dense", "sparse"] | None, typer.Option()] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
) -> None:
    """Retrieve chunks only (no LLM): useful for debugging retrieval."""
    chunks = _container().retriever.search(
        query, domains=domain, top_k=top_k, mode=mode, rerank=rerank
    )
    for i, c in enumerate(chunks, start=1):
        score = (
            f"rerank={c.rerank_score:.2f}"
            if c.rerank_score is not None
            else f"score={c.score:.3f}"
        )
        console.print(
            f"[bold]{i}. {c.breadcrumb}[/bold] [dim]({c.domain}, {score})[/dim]"
        )
        console.print(f"   [link]{c.url}[/link]")
        console.print(f"   {c.text[:240].replace(chr(10), ' ')}...\n", markup=False)


@app.command()
def diagrams(
    query: Annotated[
        str | None, typer.Argument(help="Text describing the diagram.")
    ] = None,
    image: Annotated[
        Path | None, typer.Option(help="Find diagrams similar to this image.")
    ] = None,
    limit: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Multimodal search over indexed architecture diagrams (CLIP)."""
    if (query is None) == (image is None):
        raise typer.BadParameter("give either a text QUERY or --image")
    search_ = _container().diagram_search
    results = (
        search_.by_image(image, limit=limit)
        if image
        else search_.by_text(query or "", limit=limit)
    )
    for d in results:
        console.print(f"[bold]{d.caption}[/bold] [dim](score {d.score:.3f})[/dim]")
        console.print(
            f"   image: [link]{d.url}[/link]\n   from:  [link]{d.doc_url}[/link]"
        )


@app.command("eval")
def evaluate(
    golden: Annotated[Path | None, typer.Option(help="Golden set JSON.")] = None,
    k: Annotated[int, typer.Option(min=1)] = 5,
    mode: Annotated[Literal["hybrid", "dense", "sparse"] | None, typer.Option()] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    agentic: Annotated[
        bool, typer.Option(help="Use planner + sub-queries for retrieval.")
    ] = False,
    generation: Annotated[
        bool, typer.Option(help="Also grade answers with the LLM judge.")
    ] = False,
    ablation: Annotated[
        bool, typer.Option(help="Compare dense / sparse / hybrid / +rerank.")
    ] = False,
    limit: Annotated[
        int | None, typer.Option(help="Only the first N examples.")
    ] = None,
) -> None:
    """Evaluate retrieval (and optionally generation) against the golden set."""
    from rag.evaluation import (
        evaluate_generation,
        evaluate_retrieval,
        load_golden_set,
        save_report,
    )

    container = _container()
    settings = container.settings
    examples = load_golden_set(golden or settings.paths.eval_dir / "golden_set.json")[
        :limit
    ]
    reports_dir = settings.paths.eval_dir / "reports"

    runs: list[tuple[str, dict[str, object]]] = []
    if ablation:
        runs = [
            ("dense", {"mode": "dense", "rerank": False}),
            ("bm25", {"mode": "sparse", "rerank": False}),
            ("hybrid", {"mode": "hybrid", "rerank": False}),
            ("hybrid+rerank", {"mode": "hybrid", "rerank": True}),
        ]
    else:
        runs = [("retrieval", {"mode": mode, "rerank": rerank})]

    table = Table(title=f"Retrieval quality (n={len(examples)}, k={k})")
    for col in ["run", f"hit@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", "p50 ms"]:
        table.add_column(col)
    for label, options in runs:
        report = evaluate_retrieval(
            examples,
            retriever=container.retriever,
            pipeline=container.pipeline,
            k=k,
            strategy="agentic" if agentic else "single",
            **options,  # type: ignore[arg-type]
        )
        path = save_report(report, reports_dir, label.replace("+", "_"))
        s = report["summary"]
        table.add_row(
            label,
            *(f"{s[m]:.3f}" for m in (f"hit@{k}", f"recall@{k}", "mrr", f"ndcg@{k}")),
            f"{s['latency_ms_p50']:.0f}",
        )
        console.print(f"[dim]saved {path}[/dim]")
    console.print(table)

    if generation:
        report = evaluate_generation(
            examples, pipeline=container.pipeline, judge=container.judge_llm, k=k
        )
        path = save_report(report, reports_dir, "generation")
        console.print_json(data=report["summary"])
        console.print(f"[dim]saved {path}[/dim]")


@app.command()
def stats() -> None:
    """Show what is indexed."""
    console.print_json(data=_container().store.stats())


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    reload: Annotated[
        bool, typer.Option(help="Auto-reload on code changes (dev).")
    ] = False,
) -> None:
    """Run the REST API (FastAPI + Uvicorn)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rag.api.app:create_app",
        factory=True,
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
