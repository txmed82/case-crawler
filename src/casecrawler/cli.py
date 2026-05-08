from __future__ import annotations

import asyncio
import json
import subprocess
import time

import click

from casecrawler.config import get_config, load_config
from casecrawler.pipeline.orchestrator import PipelineOrchestrator
from casecrawler.sources.registry import SourceRegistry

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


@cli.command("imaging-models")
def imaging_models() -> None:
    """List built-in synthetic medical imaging model profiles."""
    from casecrawler.generation.imaging_models import list_imaging_model_profiles

    for profile in list_imaging_model_profiles():
        click.echo(
            f"{profile.name}: {profile.model_id} "
            f"modality={profile.modality} region={profile.body_region} "
            f"license={profile.license or 'unspecified'}"
        )
        if profile.notes:
            click.echo(f"  {profile.notes}")


@cli.command("timeseries-models")
def timeseries_models() -> None:
    """List built-in EHR time-series model adapter profiles."""
    from casecrawler.generation.timeseries_models import list_time_series_model_profiles

    for profile in list_time_series_model_profiles():
        click.echo(
            f"{profile.name}: adapter={profile.adapter_type} reference={profile.reference}"
        )
        click.echo(f"  {profile.notes}")


@cli.command("reference-datasets")
def reference_datasets() -> None:
    """List configured Hugging Face reference datasets for benchmarking."""
    from casecrawler.integrations.huggingface import REFERENCE_DATASETS

    for key, spec in REFERENCE_DATASETS.items():
        click.echo(
            f"{key}: {spec.repo_id} split={spec.split} "
            f"license={spec.license}"
        )
        if spec.description:
            click.echo(f"  {spec.description}")


@cli.command("import-reference-dataset")
@click.argument("reference_key", required=False)
@click.option("--dataset-id", required=True, help="Dataset id for imported reference records")
@click.option("--repo-id", default=None, help="Custom Hugging Face dataset repo id")
@click.option("--split", default=None, help="Override the configured dataset split")
@click.option("--license", "license_name", default=None, help="License for custom repo imports")
@click.option("--note-field", default="note", help="Clinical note text field")
@click.option("--question-field", default=None, help="Optional instruction/question field")
@click.option("--answer-field", default=None, help="Optional answer/completion field")
@click.option("--task-field", default=None, help="Optional task field")
@click.option("--patient-id-field", default=None, help="Optional source patient id field")
@click.option("--limit", default=100, type=int, help="Maximum reference rows to import")
@click.option(
    "--no-streaming",
    is_flag=True,
    help="Use non-streaming Hugging Face dataset loading",
)
def import_reference_dataset(
    reference_key: str | None,
    dataset_id: str,
    repo_id: str | None,
    split: str | None,
    license_name: str | None,
    note_field: str,
    question_field: str | None,
    answer_field: str | None,
    task_field: str | None,
    patient_id_field: str | None,
    limit: int,
    no_streaming: bool,
) -> None:
    """Import a Hugging Face reference dataset into the local dataset store."""
    from casecrawler.integrations.huggingface import (
        REFERENCE_DATASETS,
        import_reference_rows,
        load_huggingface_dataset,
        load_reference_dataset,
        reference_dataset_spec,
    )
    from casecrawler.storage.dataset_store import DatasetStore

    if not reference_key and not repo_id:
        raise click.ClickException("Provide a reference key or --repo-id.")
    if not repo_id and reference_key not in REFERENCE_DATASETS:
        choices = ", ".join(sorted(REFERENCE_DATASETS))
        raise click.ClickException(
            f"Unknown reference dataset {reference_key!r}. Choose from: {choices}"
        )
    if limit < 1:
        raise click.ClickException("limit must be at least 1.")
    try:
        if repo_id:
            effective_split = split or "train"
            spec = reference_dataset_spec(
                repo_id=repo_id,
                split=effective_split,
                license=license_name or "unspecified",
                note_field=note_field,
                question_field=question_field,
                answer_field=answer_field,
                task_field=task_field,
                patient_id_field=patient_id_field,
                description="User-specified Hugging Face reference dataset.",
            )
            rows = load_huggingface_dataset(
                repo_id,
                split=effective_split,
                streaming=not no_streaming,
            )
            records = import_reference_rows(
                rows,
                dataset_id=dataset_id,
                split=effective_split,
                limit=limit,
                spec=spec,
            )
            source_name = repo_id
        else:
            assert reference_key is not None
            rows = load_reference_dataset(
                reference_key,
                split=split,
                streaming=not no_streaming,
            )
            records = import_reference_rows(
                rows,
                dataset_id=dataset_id,
                reference_key=reference_key,
                split=split,
                limit=limit,
            )
            source_name = reference_key
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    store = DatasetStore()
    for record in records:
        store.save_record(record)
    click.echo(
        f"Imported {len(records)} reference record(s) from {source_name} "
        f"into {dataset_id}"
    )


@cli.command("import-synthea-fhir")
@click.argument("path")
@click.option("--dataset-id", required=True, help="Dataset id for imported Synthea records")
def import_synthea_fhir(path: str, dataset_id: str) -> None:
    """Import Synthea FHIR JSON bundle files into the local dataset store."""
    from casecrawler.integrations.synthea import SyntheaAdapter
    from casecrawler.storage.dataset_store import DatasetStore

    try:
        records = SyntheaAdapter().import_fhir_path(path, dataset_id=dataset_id)
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Failed to import Synthea FHIR from {path}: {exc}") from exc
    if not records:
        raise click.ClickException(f"No Synthea FHIR JSON bundles found at {path}.")
    store = DatasetStore()
    for record in records:
        store.save_record(record)
    click.echo(f"Imported {len(records)} Synthea FHIR record(s) into {dataset_id}")


@cli.command("run-synthea")
@click.option("--dataset-id", required=True, help="Dataset id for imported Synthea records")
@click.option(
    "--output-dir",
    required=True,
    help="Directory where Synthea writes FHIR JSON bundles",
)
@click.option("--population", default=1, type=click.IntRange(1), show_default=True)
@click.option(
    "--synthea-executable",
    default=None,
    help="Path to run_synthea; defaults to synthetic.synthea_executable config.",
)
def run_synthea(
    dataset_id: str,
    output_dir: str,
    population: int,
    synthea_executable: str | None,
) -> None:
    """Run a configured Synthea executable and import generated FHIR bundles."""
    from casecrawler.integrations.synthea import SyntheaAdapter
    from casecrawler.storage.dataset_store import DatasetStore

    executable = synthea_executable or get_config().synthetic.synthea_executable
    if not executable:
        raise click.ClickException(
            "Provide --synthea-executable or set synthetic.synthea_executable."
        )
    try:
        records = SyntheaAdapter().run_and_import(
            executable=executable,
            output_dir=output_dir,
            dataset_id=dataset_id,
            population=population,
        )
    except (OSError, json.JSONDecodeError, ValueError, subprocess.SubprocessError) as exc:
        raise click.ClickException(f"Failed to run/import Synthea: {exc}") from exc
    if not records:
        raise click.ClickException(f"No Synthea FHIR JSON bundles found at {output_dir}.")
    store = DatasetStore()
    for record in records:
        store.save_record(record)
    click.echo(f"Ran Synthea and imported {len(records)} record(s) into {dataset_id}")


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
    "--modalities",
    default=None,
    help=(
        "Comma-separated modalities: structured_ehr,clinical_text,labs,vitals,"
        "time_series,imaging"
    ),
)
@click.option(
    "--complexity",
    default="moderate",
    type=click.Choice(["simple", "moderate", "complex", "rare"]),
    help="Synthetic record complexity profile",
)
@click.option("--age-min", default=None, type=int, help="Minimum generated patient age")
@click.option("--age-max", default=None, type=int, help="Maximum generated patient age")
@click.option("--sexes", default=None, help="Comma-separated sex cycle")
@click.option("--base-time", default=None, help="ISO-8601 base timestamp")
@click.option(
    "--clinical-text-backend",
    default=None,
    type=click.Choice(["deterministic", "llm"]),
    help="Override clinical text backend for this generation request",
)
@click.option("--llm-provider", default=None, help="Override LLM provider for clinical text")
@click.option("--llm-model", default=None, help="Override LLM model for clinical text")
@click.option(
    "--ollama-base-url",
    default=None,
    help="Override Ollama base URL for clinical text generation",
)
@click.option(
    "--imaging-backend",
    default=None,
    type=click.Choice(["placeholder", "diffusers"]),
    help="Override synthetic imaging backend for this generation request",
)
@click.option(
    "--imaging-model-profile",
    default=None,
    help="Built-in imaging model profile, for example cxr_pneumonia_dreambooth",
)
@click.option(
    "--diffusers-model-id",
    default=None,
    help="Override Hugging Face diffusers model id for this generation request",
)
@click.option(
    "--time-series-backend",
    default=None,
    type=click.Choice(["deterministic", "external"]),
    help="Override time-series backend for this generation request",
)
@click.option(
    "--time-series-model-profile",
    default=None,
    help="Built-in time-series model profile, for example timediff",
)
@click.option(
    "--time-series-command",
    default=None,
    help="Comma-separated external time-series command for this request",
)
def generate_dataset(
    topic: str,
    count: int,
    modalities: str | None,
    complexity: str,
    age_min: int | None,
    age_max: int | None,
    sexes: str | None,
    base_time: str | None,
    clinical_text_backend: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    ollama_base_url: str | None,
    imaging_backend: str | None,
    imaging_model_profile: str | None,
    diffusers_model_id: str | None,
    time_series_backend: str | None,
    time_series_model_profile: str | None,
    time_series_command: str | None,
) -> None:
    """Generate synthetic healthcare records for AI training."""
    from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
    from casecrawler.models.dataset import GenerationRequest
    from casecrawler.models.synthetic import ComplexityProfile, Modality
    from casecrawler.storage.dataset_store import DatasetStore

    complexity_profile = ComplexityProfile(complexity)
    cohort_constraints = {}
    if age_min is not None:
        cohort_constraints["age_min"] = age_min
    if age_max is not None:
        cohort_constraints["age_max"] = age_max
    if sexes:
        cohort_constraints["sexes"] = [
            value.strip() for value in sexes.split(",") if value.strip()
        ]
    if base_time:
        cohort_constraints["base_time"] = base_time
    parsed_time_series_command = (
        [value.strip() for value in time_series_command.split(",") if value.strip()]
        if time_series_command
        else None
    )
    try:
        selected_modalities = (
            [Modality(value.strip()) for value in modalities.split(",") if value.strip()]
            if modalities
            else None
        )
        req = GenerationRequest(
            topic=topic,
            count=count,
            complexity=complexity_profile,
            modalities=selected_modalities
            if selected_modalities is not None
            else GenerationRequest(topic=topic).modalities,
            cohort_constraints=cohort_constraints,
            clinical_text_backend=clinical_text_backend,
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base_url=ollama_base_url,
            imaging_backend=imaging_backend,
            imaging_model_profile=imaging_model_profile,
            diffusers_model_id=diffusers_model_id,
            time_series_backend=time_series_backend,
            time_series_model_profile=time_series_model_profile,
            time_series_command=parsed_time_series_command,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
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


@datasets_group.command("quality")
@click.argument("dataset_id")
def datasets_quality(dataset_id: str) -> None:
    """Show dataset fine-tuning export readiness."""
    from casecrawler.storage.dataset_store import DatasetStore
    from casecrawler.validation.quality import build_dataset_quality_report

    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise click.ClickException(f"Dataset {dataset_id} not found.")
    records = list(store.iter_records(dataset_id=dataset_id))
    report = build_dataset_quality_report(
        dataset_id,
        records,
        effective_approved=store.effective_approved,
    )
    click.echo(report.model_dump_json(indent=2))


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
            "tool_call_jsonl",
            "multimodal_jsonl",
            "dpo_jsonl",
            "rl_jsonl",
            "fhir_ndjson",
            "parquet",
        ]
    ),
    default="sft_jsonl",
)
@click.option("--dataset-id", default=None, help="Dataset id filter")
@click.option(
    "--allow-blocked",
    is_flag=True,
    help="Export even when dataset quality gates report blockers.",
)
def export_dataset(
    output: str,
    export_format: str,
    dataset_id: str | None,
    allow_blocked: bool,
) -> None:
    """Export synthetic datasets to fine-tuning files."""
    from casecrawler.export.fine_tuning import export_parquet_dataset, export_record
    from casecrawler.models.dataset import ExportFormat
    from casecrawler.storage.dataset_store import DatasetStore
    from casecrawler.validation.quality import build_dataset_quality_report

    store = DatasetStore()
    if dataset_id and not store.dataset_exists(dataset_id):
        raise click.ClickException(f"Dataset {dataset_id} not found.")
    records = list(store.iter_records(dataset_id=dataset_id))
    if dataset_id and not allow_blocked:
        report = build_dataset_quality_report(
            dataset_id,
            records,
            effective_approved=store.effective_approved,
        )
        if not report.export_ready:
            raise click.ClickException(
                "Dataset is not ready for fine-tuning export. "
                f"Blockers: {report.issue_counts_by_field}. "
                "Use --allow-blocked to export anyway."
            )
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
@click.option(
    "--min-overall-score",
    default=0.75,
    type=click.FloatRange(0.0, 1.0),
    show_default=True,
    help="Minimum overall benchmark score required to pass.",
)
@click.option(
    "--min-metric-score",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    show_default=True,
    help="Minimum individual metric score required to pass.",
)
@click.option("--output", default=None, help="Optional JSON report path")
def benchmark_dataset(
    dataset_id: str,
    reference_dataset_id: str,
    min_overall_score: float,
    min_metric_score: float,
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
        report = DatasetBenchmark(
            min_overall_score=min_overall_score,
            min_metric_score=min_metric_score,
        ).compare(generated_records, reference_records)
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
    click.echo(f"Passed: {str(report.passed).lower()}")
    if report.failing_metrics:
        click.echo(f"Failing metrics: {', '.join(report.failing_metrics)}")
    for metric in report.metrics:
        click.echo(f"  {metric.name}: {metric.score:.4f}")
    for warning in report.warnings:
        click.echo(f"Warning: {warning}")


@cli.command("document-dataset")
@click.option("--dataset-id", required=True, help="Dataset id")
@click.option("--output", required=True, help="Markdown output path")
@click.option(
    "--kind",
    type=click.Choice(["dataset", "model"]),
    default="dataset",
    help="Card type to generate",
)
def document_dataset(dataset_id: str, output: str, kind: str) -> None:
    """Generate a dataset or generation-pipeline card for a synthetic dataset."""
    from casecrawler.export.cards import build_dataset_card, build_model_card
    from casecrawler.storage.dataset_store import DatasetStore

    store = DatasetStore()
    try:
        manifest = store.get_manifest(dataset_id)
    except KeyError as exc:
        raise click.ClickException(f"Dataset {dataset_id} not found.") from exc
    records = list(store.iter_records(dataset_id=dataset_id))
    card = (
        build_dataset_card(manifest, records)
        if kind == "dataset"
        else build_model_card(manifest, records)
    )
    try:
        with open(output, "w") as f:
            f.write(card)
    except OSError as exc:
        raise click.ClickException(f"Failed to write {kind} card to {output}: {exc}") from exc
    click.echo(f"Wrote {kind} card for {dataset_id} to {output}")
