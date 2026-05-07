from __future__ import annotations

import asyncio
import json
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


@cli.command("generate-dataset")
@click.argument("topic")
@click.option("--count", default=1, type=int, help="Number of synthetic records to generate")
@click.option(
    "--complexity",
    default="moderate",
    type=click.Choice(["simple", "moderate", "complex", "rare"]),
    help="Synthetic record complexity profile",
)
def generate_dataset(topic: str, count: int, complexity: str) -> None:
    """Generate synthetic healthcare records for AI training."""
    from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
    from casecrawler.models.dataset import GenerationRequest
    from casecrawler.models.synthetic import ComplexityProfile
    from casecrawler.storage.dataset_store import DatasetStore

    complexity_profile = ComplexityProfile(complexity)
    req = GenerationRequest(topic=topic, count=count, complexity=complexity_profile)
    result = asyncio.run(SyntheticPipeline().generate(req))
    store = DatasetStore()
    for record in result["records"]:
        store.save_record(record)
    click.echo(f"Dataset: {result['dataset_id']}")
    click.echo(f"Generated: {result['generated']}")
    click.echo(f"Approved: {result['approved']}")


@cli.group("datasets", invoke_without_command=True)
@click.pass_context
def datasets_group(ctx: click.Context) -> None:
    """Manage synthetic healthcare datasets."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(datasets_list)


@datasets_group.command("list")
@click.option("--limit", default=100, type=int, help="Max datasets")
def datasets_list(limit: int) -> None:
    """List synthetic datasets."""
    from casecrawler.storage.dataset_store import DatasetStore

    manifests = DatasetStore().list_manifests(limit=limit)
    if not manifests:
        click.echo("No datasets found.")
        return
    for manifest in manifests:
        click.echo(
            f"{manifest.dataset_id} {manifest.topic} "
            f"generated={manifest.generated_count} approved={manifest.approved_count}"
        )


@datasets_group.command("show")
@click.argument("dataset_id")
def datasets_show(dataset_id: str) -> None:
    """Show a synthetic dataset manifest."""
    from casecrawler.storage.dataset_store import DatasetStore

    store = DatasetStore()
    try:
        manifest = store.get_manifest(dataset_id)
    except KeyError:
        click.echo(f"Dataset {dataset_id} not found.")
        return
    click.echo(manifest.model_dump_json(indent=2))


@cli.command("validate")
@click.option("--dataset-id", default=None, help="Dataset id prefix or exact id")
def validate_dataset(dataset_id: str | None) -> None:
    """Re-run validation for stored synthetic records."""
    from casecrawler.storage.dataset_store import DatasetStore
    from casecrawler.validation.synthetic_validator import SyntheticValidator

    store = DatasetStore()
    validator = SyntheticValidator()
    approved = 0
    validated = 0
    for record in store.iter_records():
        if dataset_id and not record.dataset_id.startswith(dataset_id):
            continue
        validation = validator.validate(record)
        validated += 1
        if validation.approved:
            approved += 1
        store.save_record(record.model_copy(update={"validation": validation}))
    click.echo(f"Validated: {validated}")
    click.echo(f"Approved: {approved}")


@cli.group("reviews", invoke_without_command=True)
@click.pass_context
def reviews_group(ctx: click.Context) -> None:
    """Manage human review decisions for synthetic records."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(reviews_queue)


@reviews_group.command("queue")
@click.option("--dataset-id", default=None, help="Dataset id filter")
@click.option("--limit", default=100, type=int, help="Max records")
@click.option("--include-reviewed", is_flag=True, help="Include closed review items")
def reviews_queue(dataset_id: str | None, limit: int, include_reviewed: bool) -> None:
    """List records that need human review."""
    from casecrawler.storage.dataset_store import DatasetStore

    store = DatasetStore()
    items = store.list_review_queue(
        dataset_id=dataset_id,
        include_reviewed=include_reviewed,
        limit=limit,
    )
    if not items:
        click.echo("No records need human review.")
        return
    for item in items:
        review_status = item.human_review.status.value if item.human_review else "pending"
        click.echo(
            f"{item.record_id} {item.dataset_id} {item.topic} "
            f"validation={item.validation_approved} review={review_status} "
            f"issues={item.issue_count}"
        )


@reviews_group.command("mark")
@click.argument("record_id")
@click.option(
    "--status",
    "review_status",
    required=True,
    type=click.Choice(["approved", "rejected", "needs_revision", "pending"]),
)
@click.option("--reviewer", default="human", help="Reviewer identifier")
@click.option("--note", "notes", multiple=True, help="Reviewer note")
def reviews_mark(
    record_id: str,
    review_status: str,
    reviewer: str,
    notes: tuple[str, ...],
) -> None:
    """Save a human review decision for a synthetic record."""
    from casecrawler.models.dataset import HumanReviewDecision, HumanReviewStatus
    from casecrawler.storage.dataset_store import DatasetStore

    store = DatasetStore()
    try:
        record = store.save_human_review(
            record_id,
            HumanReviewDecision(
                status=HumanReviewStatus(review_status),
                reviewer=reviewer,
                notes=list(notes),
            ),
        )
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Reviewed {record.record_id}: "
        f"{record.metadata['human_review']['status']} "
        f"effective_approved={store.effective_approved(record)}"
    )


@cli.command("export-dataset")
@click.option("--output", required=True, help="Output file path")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(
        [
            "raw_jsonl",
            "sft_jsonl",
            "chat_jsonl",
            "multimodal_jsonl",
            "fhir_ndjson",
            "parquet",
        ]
    ),
    default="sft_jsonl",
)
@click.option("--dataset-id", default=None, help="Dataset id filter")
def export_dataset(output: str, export_format: str, dataset_id: str | None) -> None:
    """Export synthetic datasets to fine-tuning files."""
    from casecrawler.export.fine_tuning import export_parquet_dataset, export_record
    from casecrawler.models.dataset import ExportFormat
    from casecrawler.storage.dataset_store import DatasetStore

    store = DatasetStore()
    if dataset_id and not store.dataset_exists(dataset_id):
        raise click.ClickException(f"Dataset {dataset_id} not found.")
    records = store.iter_records(dataset_id=dataset_id)
    if ExportFormat(export_format) == ExportFormat.PARQUET:
        try:
            record_count = export_parquet_dataset(records, output)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        record_count = 0
        with open(output, "w") as f:
            for record in records:
                f.write(json.dumps(export_record(record, export_format), sort_keys=True) + "\n")
                record_count += 1
    if dataset_id:
        store.save_export_manifest(
            dataset_id=dataset_id,
            export_format=export_format,
            file_path=output,
            record_count=record_count,
        )
    click.echo(f"Exported {record_count} record(s) to {output}")


@cli.command("benchmark-dataset")
@click.option("--dataset-id", required=True, help="Generated dataset id")
@click.option("--reference-dataset-id", required=True, help="Reference dataset id")
@click.option("--output", default=None, help="Optional JSON report path")
def benchmark_dataset(
    dataset_id: str,
    reference_dataset_id: str,
    output: str | None,
) -> None:
    """Compare a generated dataset against a stored reference dataset."""
    from casecrawler.storage.dataset_store import DatasetStore
    from casecrawler.validation.benchmark import DatasetBenchmark

    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise click.ClickException(f"Dataset {dataset_id} not found.")
    if not store.dataset_exists(reference_dataset_id):
        raise click.ClickException(f"Reference dataset {reference_dataset_id} not found.")
    generated_records = list(store.iter_records(dataset_id=dataset_id))
    reference_records = list(store.iter_records(dataset_id=reference_dataset_id))
    try:
        report = DatasetBenchmark().compare(generated_records, reference_records)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    payload = report.model_dump_json(indent=2)
    if output:
        try:
            with open(output, "w") as f:
                f.write(payload + "\n")
        except OSError as exc:
            raise click.ClickException(
                f"Failed to write benchmark report to {output}: {exc}"
            ) from exc
    click.echo(f"Benchmark: {report.generated_dataset_id} vs {report.reference_dataset_id}")
    click.echo(f"Overall score: {report.overall_score:.4f}")
    for metric in report.metrics:
        click.echo(f"  {metric.name}: {metric.score:.4f}")
    for warning in report.warnings:
        click.echo(f"Warning: {warning}")


@cli.command()
@click.argument("topic")
@click.option("--difficulty", default=None, help="medical_student, resident, or attending")
@click.option("--count", default=1, type=int, help="Number of cases to generate")
@click.option("--ingest", "ingest_first", is_flag=True, help="Ingest topic first")
@click.option("--output", default=None, help="Output JSONL file path")
@click.option("--multi-step", "multi_step", is_flag=True, help="Generate multi-step cases with structured diagnostics")
def generate(topic: str, difficulty: str | None, count: int, ingest_first: bool, output: str | None, multi_step: bool) -> None:
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
    if multi_step:
        from casecrawler.generation.multi_step_pipeline import MultiStepPipeline
        gen_pipeline = MultiStepPipeline(
            provider=provider,
            retriever=retriever,
            max_retries=config.generation.max_retries,
            review_threshold=config.generation.review_threshold,
        )
    else:
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


@cli.command()
@click.argument("output_path")
@click.option("--format", "export_format", type=click.Choice(["rl", "sft", "both"]), default="both")
@click.option("--difficulty", default=None, help="Filter by difficulty")
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--min-accuracy", default=None, type=float, help="Minimum accuracy score")
@click.option("--include-wrong-paths", is_flag=True, help="Include wrong-path SFT variants")
def export(
    output_path: str,
    export_format: str,
    difficulty: str | None,
    topic: str | None,
    min_accuracy: float | None,
    include_wrong_paths: bool,
) -> None:
    """Export multi-step cases to training data formats."""
    import json

    from casecrawler.export.rl_exporter import export_rl_episode
    from casecrawler.export.sft_exporter import export_sft_conversation

    case_store = CaseStore()
    cases = case_store.list_cases(topic=topic, difficulty=difficulty, min_accuracy=min_accuracy, limit=10000)
    multi_step_cases = [c for c in cases if c.is_multi_step()]

    if not multi_step_cases:
        click.echo("No multi-step cases found matching filters.")
        return

    exported = 0
    with open(output_path, "w") as f:
        for case in multi_step_cases:
            if export_format in ("rl", "both"):
                episode = export_rl_episode(case)
                f.write(json.dumps({"type": "rl_episode", **episode.model_dump()}) + "\n")
                exported += 1
            if export_format in ("sft", "both"):
                conv = export_sft_conversation(case)
                f.write(json.dumps({"type": "sft_conversation", **conv.model_dump()}) + "\n")
                exported += 1
                if include_wrong_paths:
                    wrong = export_sft_conversation(case, include_wrong_path=True)
                    f.write(json.dumps({"type": "sft_wrong_path", **wrong.model_dump()}) + "\n")
                    exported += 1

    click.echo(f"Exported {exported} records from {len(multi_step_cases)} cases to {output_path}")
