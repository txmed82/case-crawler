from __future__ import annotations

import asyncio
import time

import click

from casecrawler.config import get_config, load_config
from casecrawler.generation.pipeline import GenerationPipeline
from casecrawler.generation.retriever import Retriever
from casecrawler.llm.factory import get_provider
from casecrawler.pipeline.orchestrator import PipelineOrchestrator
from casecrawler.pipeline.store import Store
from casecrawler.sources.registry import SourceRegistry
from casecrawler.storage.case_store import CaseStore

# Import all source modules so BaseSource.__subclasses__() discovers them
import casecrawler.sources.pubmed  # noqa: F401
import casecrawler.sources.openfda  # noqa: F401
import casecrawler.sources.dailymed  # noqa: F401
import casecrawler.sources.rxnorm  # noqa: F401
import casecrawler.sources.medrxiv  # noqa: F401
import casecrawler.sources.clinicaltrials  # noqa: F401
import casecrawler.sources.glass  # noqa: F401
import casecrawler.sources.annas_archive  # noqa: F401
import casecrawler.sources.firecrawl  # noqa: F401


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
def cli(config_path: str | None) -> None:
    """CaseCrawler — Medical knowledge ingestion engine."""
    load_config(config_path)


@cli.command()
@click.argument("query")
@click.option("--sources", default=None, help="Comma-separated source names")
@click.option("--limit", default=None, type=int, help="Max results per source")
def ingest(query: str, sources: str | None, limit: int | None) -> None:
    """Ingest medical content for a topic from available sources."""
    config = get_config()
    limit = limit or config.ingestion.default_limit_per_source

    registry = SourceRegistry()
    registry.discover()

    source_names = sources.split(",") if sources else None
    active_sources = registry.get_sources(source_names)

    if not active_sources:
        click.echo("No sources available. Check your API keys with 'casecrawler sources'.")
        return

    click.echo(f"Ingesting '{query}' from {len(active_sources)} source(s)...")
    start = time.time()

    # Fan out searches in parallel
    all_docs = asyncio.run(_search_all(active_sources, query, limit))

    # Process through pipeline
    pipeline = PipelineOrchestrator()
    total_summary: dict[str, dict] = {}

    for source_name, docs in all_docs.items():
        if docs:
            result = pipeline.process(docs)
            total_summary[source_name] = result

    elapsed = time.time() - start

    # Print summary
    click.echo("\n--- Ingestion Summary ---")
    total_docs = 0
    total_chunks = 0
    for source_name, summary in total_summary.items():
        click.echo(f"  {source_name}: {summary['documents']} documents, {summary['chunks']} chunks")
        total_docs += summary["documents"]
        total_chunks += summary["chunks"]
    click.echo(f"\nTotal: {total_docs} documents, {total_chunks} chunks in {elapsed:.1f}s")


async def _search_all(
    sources: list, query: str, limit: int
) -> dict[str, list]:
    """Fan out search calls to all sources concurrently."""

    async def _search_one(source):
        try:
            docs = await source.search(query, limit=limit)
            return source.name, docs
        except Exception as e:
            click.echo(f"  Warning: {source.name} failed: {e}")
            return source.name, []

    tasks = [_search_one(s) for s in sources]
    results = await asyncio.gather(*tasks)
    return dict(results)


@cli.command()
@click.argument("query")
@click.option("--source", default=None, help="Filter by source name")
@click.option("--limit", default=10, type=int, help="Max results")
def search(query: str, source: str | None, limit: int) -> None:
    """Search the knowledge base."""
    pipeline = PipelineOrchestrator()
    results = pipeline.store.search(query, n_results=limit, source=source)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score = r["score"]
        meta = r["metadata"]
        text_preview = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        click.echo(f"\n[{i}] (score: {score:.3f}) [{meta.get('source', '?')}] [{meta.get('credibility', '?')}]")
        click.echo(f"    {text_preview}")


@cli.command()
def sources() -> None:
    """List available and unavailable data sources."""
    registry = SourceRegistry()
    info = registry.all_sources_info()

    available = [s for s in info if s["available"]]
    unavailable = [s for s in info if not s["available"]]

    click.echo("Available:")
    for s in available:
        keys_info = ", ".join(s["requires_keys"]) if s["requires_keys"] else "no key required"
        click.echo(f"  \u2713 {s['name']:<18} ({keys_info})")

    if unavailable:
        click.echo("\nUnavailable:")
        for s in unavailable:
            missing = ", ".join(s.get("missing_keys", []))
            click.echo(f"  \u2717 {s['name']:<18} (missing {missing})")


@cli.command("config")
def show_config() -> None:
    """Show current configuration."""
    config = get_config()
    click.echo(f"Ingestion limit per source: {config.ingestion.default_limit_per_source}")
    click.echo(f"Chunk size: {config.chunking.default_chunk_size}")
    click.echo(f"Chunk overlap: {config.chunking.overlap}")
    click.echo(f"Embedding model: {config.embedding.model}")
    click.echo(f"ChromaDB dir: {config.storage.chroma_persist_dir}")
    click.echo(f"API: {config.api.host}:{config.api.port}")


@cli.command()
def serve() -> None:
    """Start the FastAPI server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "casecrawler.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
    )


@cli.command()
@click.argument("topic")
@click.option("--difficulty", default=None, help="medical_student, resident, or attending")
@click.option("--count", default=1, type=int, help="Number of cases to generate")
@click.option("--ingest", "ingest_first", is_flag=True, help="Ingest topic first")
@click.option("--output", default=None, help="Output JSONL file path")
def generate(topic: str, difficulty: str | None, count: int, ingest_first: bool, output: str | None) -> None:
    """Generate clinical cases for a medical topic."""
    config = get_config()
    difficulty = difficulty or config.generation.default_difficulty

    if ingest_first:
        click.echo(f"Ingesting '{topic}' first...")
        registry = SourceRegistry()
        registry.discover()
        active_sources = registry.get_sources()
        if active_sources:
            all_docs = asyncio.run(_search_all(active_sources, topic, config.ingestion.default_limit_per_source))
            pipeline_orch = PipelineOrchestrator()
            for source_name, docs in all_docs.items():
                if docs:
                    pipeline_orch.process(docs)

    # Check ChromaDB has content
    store = Store(chroma_dir=config.storage.chroma_persist_dir)
    if store.count == 0:
        click.echo(f"No content found for '{topic}'. Run 'casecrawler ingest \"{topic}\"' first.")
        return

    try:
        provider = get_provider(config.llm.provider, config.llm.model, base_url=config.llm.ollama_base_url)
    except ValueError as e:
        click.echo(f"Error: {e}")
        return

    retriever = Retriever(store=store)
    gen_pipeline = GenerationPipeline(
        provider=provider,
        retriever=retriever,
        max_retries=config.generation.max_retries,
        review_threshold=config.generation.review_threshold,
    )

    click.echo(f"Generating {count} case(s) for '{topic}' at {difficulty} difficulty...")
    start = time.time()

    result = asyncio.run(gen_pipeline.generate_batch(topic=topic, count=count, difficulty=difficulty))
    elapsed = time.time() - start

    # Save to SQLite
    case_store = CaseStore()
    for case in result["cases"]:
        case_store.save(case)

    click.echo("\n--- Generation Summary ---")
    click.echo(f"  Generated: {result['generated']}")
    click.echo(f"  Failed: {result['failed']}")
    click.echo(f"  Tokens: {result['total_input_tokens']} in / {result['total_output_tokens']} out")
    click.echo(f"  Time: {elapsed:.1f}s")

    if output and result["cases"]:
        with open(output, "w") as f:
            for case in result["cases"]:
                f.write(case.model_dump_json() + "\n")
        click.echo(f"  Exported to: {output}")


@cli.group(invoke_without_command=True)
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--difficulty", default=None, help="Filter by difficulty")
@click.option("--limit", default=20, type=int, help="Max results")
@click.pass_context
def cases(ctx: click.Context, topic: str | None, difficulty: str | None, limit: int) -> None:
    """Manage generated cases. With no subcommand, lists cases."""
    if ctx.invoked_subcommand is None:
        case_store = CaseStore()
        results = case_store.list_cases(topic=topic, difficulty=difficulty, limit=limit)
        if not results:
            click.echo("No cases found.")
            return
        click.echo(f"Found {len(results)} case(s):\n")
        for case in results:
            acc = case.review.accuracy_score if case.review else 0
            click.echo(f"  [{case.case_id[:8]}] {case.topic} ({case.difficulty.value}) — accuracy: {acc:.2f}")


@cases.command("list")
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--difficulty", default=None, help="Filter by difficulty")
@click.option("--limit", default=20, type=int, help="Max results")
def cases_list(topic: str | None, difficulty: str | None, limit: int) -> None:
    """List generated cases."""
    case_store = CaseStore()
    results = case_store.list_cases(topic=topic, difficulty=difficulty, limit=limit)

    if not results:
        click.echo("No cases found.")
        return

    click.echo(f"Found {len(results)} case(s):\n")
    for case in results:
        acc = case.review.accuracy_score if case.review else 0
        click.echo(f"  [{case.case_id[:8]}] {case.topic} ({case.difficulty.value}) — accuracy: {acc:.2f}")


@cases.command("show")
@click.argument("case_id")
def cases_show(case_id: str) -> None:
    """Show a single case."""
    case_store = CaseStore()
    case = case_store.get(case_id)
    if case is None:
        click.echo(f"Case {case_id} not found.")
        return
    click.echo(case.model_dump_json(indent=2))


@cases.command("export")
@click.option("--output", required=True, help="Output JSONL file path")
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--difficulty", default=None, help="Filter by difficulty")
def cases_export(output: str, topic: str | None, difficulty: str | None) -> None:
    """Export cases to JSONL."""
    case_store = CaseStore()
    lines = case_store.export_jsonl(topic=topic, difficulty=difficulty)
    with open(output, "w") as f:
        for line in lines:
            f.write(line + "\n")
    click.echo(f"Exported {len(lines)} case(s) to {output}")
