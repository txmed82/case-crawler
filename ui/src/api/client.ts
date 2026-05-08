// TypeScript types matching the FastAPI backend models

export type CredibilityLevel = 'high' | 'medium' | 'low' | 'unknown';

export interface Source {
  id: string;
  name: string;
  description: string;
  available: boolean;
  requires_api_key: boolean;
  credibility_level: CredibilityLevel;
}

export interface SourcesResponse {
  sources: Source[];
}

export interface IngestRequest {
  topic: string;
  source_ids?: string[];
  max_results?: number;
}

export interface IngestResponse {
  job_id: string;
  status: string;
  message: string;
}

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface IngestStatusResponse {
  job_id: string;
  status: JobStatus;
  topic: string;
  progress: number;
  total: number;
  chunks_stored: number;
  errors: string[];
  created_at: string;
  completed_at?: string;
}

export interface ChunkResult {
  id: string;
  content: string;
  source_id: string;
  source_name: string;
  credibility_level: CredibilityLevel;
  document_title: string;
  document_url?: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  query: string;
  results: ChunkResult[];
  total: number;
}

// API fetch functions — all use /api prefix (proxied by Vite to localhost:8000)

const BASE = '/api';

export async function fetchSources(): Promise<SourcesResponse> {
  const res = await fetch('/api/sources');
  if (!res.ok) throw new Error(`Failed to fetch sources: ${res.statusText}`);
  return res.json();
}

export async function startIngest(request: IngestRequest): Promise<IngestResponse> {
  const res = await fetch('/api/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(`Failed to start ingest: ${res.statusText}`);
  return res.json();
}

export async function getIngestStatus(jobId: string): Promise<IngestStatusResponse> {
  const res = await fetch(`/api/ingest/${jobId}`);
  if (!res.ok) throw new Error(`Failed to get ingest status: ${res.statusText}`);
  return res.json();
}

export async function searchChunks(
  query: string,
  sourceIds?: string[],
  nResults?: number
): Promise<SearchResponse> {
  const params = new URLSearchParams({ query });
  if (sourceIds && sourceIds.length > 0) {
    sourceIds.forEach((id) => params.append('source_ids', id));
  }
  if (nResults !== undefined) {
    params.set('n_results', String(nResults));
  }
  const res = await fetch(`/api/search?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to search: ${res.statusText}`);
  return res.json();
}

export interface DatasetGenerateRequest {
  topic: string;
  count?: number;
  recipe?: string;
  complexity?: "simple" | "moderate" | "complex" | "rare";
  modalities?: SyntheticModality[];
  export_formats?: ExportFormat[];
  clinical_text_backend?: "deterministic" | "llm";
  llm_provider?: string;
  llm_model?: string;
  ollama_base_url?: string;
  imaging_backend?: "placeholder" | "diffusers";
  imaging_model_profile?: string;
  diffusers_model_id?: string;
  time_series_backend?: "deterministic" | "external";
  time_series_model_profile?: string;
  time_series_command?: string[];
  cohort_constraints?: Record<string, unknown>;
}

export interface DatasetGenerateResponse {
  dataset_id: string;
  generated: number;
  approved: number;
  total_records: number;
  records: Record<string, unknown>[];
}

export interface ReferenceDatasetCatalogItem {
  key: string;
  repo_id: string;
  split: string;
  license: string;
  description: string;
  image_field?: string | null;
  image_label_field?: string | null;
  lab_values_field?: string | null;
  vital_values_field?: string | null;
  medications_field?: string | null;
  time_series_field?: string | null;
  image_modality: string;
  image_body_region: string;
  gated: boolean;
  use_policy: string;
}

export interface ReferenceDatasetCatalogResponse {
  datasets: ReferenceDatasetCatalogItem[];
}

export interface ReferenceDatasetImportRequest {
  reference_key?: string;
  dataset_id: string;
  repo_id?: string;
  split?: string;
  license?: string;
  note_field?: string;
  question_field?: string;
  answer_field?: string;
  task_field?: string;
  patient_id_field?: string;
  image_field?: string;
  image_label_field?: string;
  image_modality?: string;
  image_body_region?: string;
  lab_values_field?: string;
  vital_values_field?: string;
  medications_field?: string;
  time_series_field?: string;
  limit?: number;
}

export interface ReferenceDatasetImportResponse {
  dataset_id: string;
  imported: number;
  reference_key: string;
  repo_id: string;
  split: string;
  license: string;
}

export interface SyntheaImportRequest {
  path: string;
  dataset_id: string;
}

export interface SyntheaImportResponse {
  dataset_id: string;
  imported: number;
  source: string;
}

export interface DatasetCapabilitiesResponse {
  modalities: SyntheticModality[];
  complexity_profiles: Array<"simple" | "moderate" | "complex" | "rare">;
  export_formats: ExportFormat[];
  generation_recipes: Array<{
    name: string;
    description: string;
    complexity: "simple" | "moderate" | "complex" | "rare";
    modalities: SyntheticModality[];
    export_formats: ExportFormat[];
    cohort_constraints: Record<string, unknown>;
    validation_threshold: number;
    recommended_reference_keys: string[];
    benchmark_thresholds: {
      min_overall_score: number;
      min_metric_score: number;
    };
  }>;
  cohort_constraints: string[];
  imaging_model_profiles: Array<{
    name: string;
    model_id: string;
    modality: string;
    body_region: string;
    license?: string | null;
    notes: string;
  }>;
  time_series_model_profiles: Array<{
    name: string;
    adapter_type: string;
    reference: string;
    notes: string;
  }>;
  clinical_profiles: Array<{
    key: string;
    keywords: string[];
    diagnosis_display: string;
    diagnosis_code: string;
    lab_names: string[];
    vital_names: string[];
    medication_names: string[];
  }>;
  validators: Array<{
    key: string;
    requires: string[];
    description: string;
  }>;
}

export type SyntheticModality =
  | "structured_ehr"
  | "clinical_text"
  | "labs"
  | "vitals"
  | "time_series"
  | "imaging";

export type ExportFormat =
  | "raw_jsonl"
  | "sft_jsonl"
  | "note_fact_sft_jsonl"
  | "medication_reconciliation_jsonl"
  | "chat_jsonl"
  | "tool_call_jsonl"
  | "multimodal_jsonl"
  | "time_series_jsonl"
  | "dpo_jsonl"
  | "rl_jsonl"
  | "fhir_ndjson"
  | "parquet";

export interface DatasetManifest {
  dataset_id: string;
  name: string;
  topic: string;
  requested_count: number;
  generated_count: number;
  approved_count: number;
  modalities: SyntheticModality[];
  export_formats: ExportFormat[];
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface ClinicalDocument {
  document_id: string;
  note_type: string;
  author_role: string;
  timestamp: string;
  clean_text: string;
  messy_text?: string | null;
  extracted_facts: Record<string, unknown>;
}

export interface SyntheticRecordPreview {
  record_id: string;
  dataset_id: string;
  topic: string;
  complexity: string;
  modalities: SyntheticModality[];
  patient: {
    patient_id: string;
    age: number;
    sex: string;
    demographics: Record<string, unknown>;
    social_history: Record<string, unknown>;
  };
  labs: { name: string; value: number | string; unit: string; flag?: string | null }[];
  vitals: { name: string; value: number; unit: string }[];
  medication_history: {
    name: string;
    dose?: string | null;
    route?: string | null;
    frequency?: string | null;
    status: string;
  }[];
  documents: ClinicalDocument[];
  imaging: { image_id: string; modality: string; body_region: string; file_path?: string | null }[];
  validation?: {
    schema_score: number;
    clinical_consistency_score: number;
    privacy_score: number;
    utility_score: number;
    modality_alignment_score?: number | null;
    approved: boolean;
  } | null;
}

export interface DatasetListResponse {
  datasets: DatasetManifest[];
}

export interface DatasetDetailResponse {
  manifest: DatasetManifest;
  records: SyntheticRecordPreview[];
}

export interface ExportManifest {
  dataset_id: string;
  export_format: ExportFormat;
  file_path: string;
  record_count: number;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface ExportManifestResponse {
  dataset_id: string;
  exports: ExportManifest[];
}

export type HumanReviewStatus =
  | "pending"
  | "approved"
  | "rejected"
  | "needs_revision";

export interface HumanReviewDecision {
  status: HumanReviewStatus;
  reviewer?: string;
  notes?: string[];
  reviewed_at?: string;
  metadata?: Record<string, unknown>;
}

export interface ReviewQueueItem {
  record_id: string;
  dataset_id: string;
  topic: string;
  complexity: string;
  modalities: SyntheticModality[];
  validation_approved?: boolean | null;
  human_review?: HumanReviewDecision | null;
  issue_count: number;
  blocking_issue_count: number;
}

export interface ReviewQueueResponse {
  dataset_id: string;
  records: ReviewQueueItem[];
}

export interface ReviewSaveResponse {
  record_id: string;
  dataset_id: string;
  human_review: HumanReviewDecision;
  effective_approved: boolean;
}

export interface BenchmarkMetric {
  name: string;
  score: number;
  generated_value: number | string | null;
  reference_value: number | string | null;
  details: Record<string, unknown>;
}

export interface BenchmarkReport {
  generated_dataset_id: string;
  reference_dataset_id: string;
  overall_score: number;
  passed: boolean;
  failing_metrics: string[];
  thresholds: Record<string, number>;
  metrics: BenchmarkMetric[];
  warnings: string[];
}

export interface BenchmarkPlanReadiness {
  dataset_id: string;
  primary_recipe?: string | null;
  recommended_reference_keys: string[];
  resolved_reference_dataset_id?: string | null;
  resolved_reference_key?: string | null;
  ready: boolean;
  missing_reference_keys: string[];
  thresholds: {
    min_overall_score: number;
    min_metric_score: number;
  };
}

export interface BenchmarkSuiteResult {
  reference_key: string;
  reference_dataset_id: string;
  passed: boolean;
  overall_score: number;
  failing_metrics: string[];
  report: BenchmarkReport;
}

export interface BenchmarkSuiteReport {
  dataset_id: string;
  primary_recipe?: string | null;
  recommended_reference_keys: string[];
  reference_count: number;
  passed: boolean;
  mean_overall_score: number;
  thresholds: {
    min_overall_score: number;
    min_metric_score: number;
  };
  results: BenchmarkSuiteResult[];
}

export interface DatasetQualityReport {
  dataset_id: string;
  record_count: number;
  approved_count: number;
  approval_rate: number;
  export_ready: boolean;
  benchmark_ready?: boolean | null;
  recommended_reference_keys: string[];
  resolved_reference_dataset_id?: string | null;
  missing_reference_keys: string[];
  benchmark_thresholds: Record<string, number>;
  modality_counts: Record<string, number>;
  artifact_counts: Record<string, number>;
  longitudinal_record_rate?: number | null;
  mean_encounter_span_hours?: number | null;
  mean_observations_per_encounter?: number | null;
  note_type_counts: Record<string, number>;
  extracted_fact_key_counts: Record<string, number>;
  lab_numeric_summaries: Record<string, Record<string, number>>;
  vital_numeric_summaries: Record<string, Record<string, number>>;
  diagnosis_code_system_counts: Record<string, number>;
  diagnosis_code_counts: Record<string, number>;
  phi_entity_counts: Record<string, number>;
  time_series_backend_counts: Record<string, number>;
  time_series_numeric_summaries: Record<string, Record<string, number>>;
  imaging_backend_counts: Record<string, number>;
  imaging_model_policy_counts: Record<string, number>;
  mean_modality_alignment_score?: number | null;
  blocking_issue_count: number;
  warning_issue_count: number;
  issue_counts_by_field: Record<string, number>;
  recommendations: string[];
}

async function readApiError(resp: Response): Promise<string> {
  try {
    const body = await resp.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((item: { msg?: string } | string) =>
          typeof item === "string" ? item : item.msg ?? String(item)
        )
        .join(", ");
    }
  } catch {
    // Fall through to status text below.
  }
  return resp.statusText || `HTTP ${resp.status}`;
}

export async function startDatasetGenerate(
  req: DatasetGenerateRequest
): Promise<DatasetGenerateResponse> {
  const resp = await fetch(`${BASE}/datasets/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) throw new Error(`Failed to generate dataset: ${await readApiError(resp)}`);
  return resp.json();
}

export async function fetchReferenceDatasetCatalog(): Promise<ReferenceDatasetCatalogResponse> {
  const resp = await fetch(`${BASE}/datasets/reference-catalog`);
  if (!resp.ok) throw new Error(`Failed to fetch reference datasets: ${await readApiError(resp)}`);
  return resp.json();
}

export async function importReferenceDataset(
  req: ReferenceDatasetImportRequest
): Promise<ReferenceDatasetImportResponse> {
  const resp = await fetch(`${BASE}/datasets/reference-import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) throw new Error(`Failed to import reference dataset: ${await readApiError(resp)}`);
  return resp.json();
}

export async function importSyntheaFhir(
  req: SyntheaImportRequest
): Promise<SyntheaImportResponse> {
  const resp = await fetch(`${BASE}/datasets/synthea-import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) throw new Error(`Failed to import Synthea FHIR: ${await readApiError(resp)}`);
  return resp.json();
}

export async function fetchDatasetCapabilities(): Promise<DatasetCapabilitiesResponse> {
  const resp = await fetch(`${BASE}/datasets/capabilities`);
  if (!resp.ok) throw new Error(`Failed to fetch dataset capabilities: ${await readApiError(resp)}`);
  return resp.json();
}

export async function fetchDatasets(limit = 100): Promise<DatasetListResponse> {
  const resp = await fetch(`${BASE}/datasets?limit=${limit}`);
  if (!resp.ok) throw new Error(`Failed to fetch datasets: ${await readApiError(resp)}`);
  return resp.json();
}

export async function fetchDataset(datasetId: string, limit = 25): Promise<DatasetDetailResponse> {
  const resp = await fetch(`${BASE}/datasets/${datasetId}?limit=${limit}`);
  if (!resp.ok) throw new Error(`Failed to fetch dataset: ${await readApiError(resp)}`);
  return resp.json();
}

export async function fetchDatasetExports(
  datasetId: string,
  limit = 25
): Promise<ExportManifestResponse> {
  const resp = await fetch(`${BASE}/datasets/${datasetId}/exports?limit=${limit}`);
  if (!resp.ok) throw new Error(`Failed to fetch export audit trail: ${await readApiError(resp)}`);
  return resp.json();
}

export async function fetchDatasetReviewQueue(
  datasetId: string,
  includeReviewed = false,
  limit = 100
): Promise<ReviewQueueResponse> {
  const qs = new URLSearchParams({
    include_reviewed: String(includeReviewed),
    limit: String(limit),
  });
  const resp = await fetch(`${BASE}/datasets/${datasetId}/reviews?${qs}`);
  if (!resp.ok) throw new Error(`Failed to fetch review queue: ${await readApiError(resp)}`);
  return resp.json();
}

export async function saveRecordReview(
  recordId: string,
  decision: HumanReviewDecision
): Promise<ReviewSaveResponse> {
  const resp = await fetch(`${BASE}/records/${recordId}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(decision),
  });
  if (!resp.ok) throw new Error(`Failed to save review: ${await readApiError(resp)}`);
  return resp.json();
}

export async function fetchDatasetCard(
  datasetId: string,
  kind: "dataset" | "model" = "dataset"
): Promise<string> {
  const qs = new URLSearchParams({ kind });
  const resp = await fetch(`${BASE}/datasets/${datasetId}/card?${qs}`);
  if (!resp.ok) throw new Error(`Failed to fetch ${kind} card: ${await readApiError(resp)}`);
  return resp.text();
}

export async function fetchDatasetBenchmark(
  datasetId: string,
  referenceDatasetId: string,
  minOverallScore = 0.75,
  minMetricScore = 0.5
): Promise<BenchmarkReport> {
  const qs = new URLSearchParams({
    reference_dataset_id: referenceDatasetId,
    min_overall_score: String(minOverallScore),
    min_metric_score: String(minMetricScore),
  });
  const resp = await fetch(`${BASE}/datasets/${datasetId}/benchmark?${qs}`);
  if (!resp.ok) throw new Error(`Failed to benchmark dataset: ${await readApiError(resp)}`);
  return resp.json();
}

export async function fetchDatasetBenchmarkPlan(
  datasetId: string
): Promise<BenchmarkPlanReadiness> {
  const resp = await fetch(`${BASE}/datasets/${datasetId}/benchmark-plan`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch benchmark plan: ${await readApiError(resp)}`);
  }
  return resp.json();
}

export async function fetchDatasetBenchmarkSuite(
  datasetId: string
): Promise<BenchmarkSuiteReport> {
  const resp = await fetch(`${BASE}/datasets/${datasetId}/benchmark-suite`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch benchmark suite: ${await readApiError(resp)}`);
  }
  return resp.json();
}

export async function fetchDatasetQuality(datasetId: string): Promise<DatasetQualityReport> {
  const resp = await fetch(`${BASE}/datasets/${datasetId}/quality`);
  if (!resp.ok) throw new Error(`Failed to fetch dataset quality: ${await readApiError(resp)}`);
  return resp.json();
}

export function datasetExportUrl(
  datasetId: string,
  exportFormat: ExportFormat = "sft_jsonl",
  benchmarkGate?: {
    referenceDatasetId?: string;
    autoBenchmark?: boolean;
    minOverallScore?: number;
    minMetricScore?: number;
  }
): string {
  const qs = new URLSearchParams({ export_format: exportFormat });
  if (benchmarkGate?.referenceDatasetId) {
    qs.set("reference_dataset_id", benchmarkGate.referenceDatasetId);
  }
  if (benchmarkGate?.autoBenchmark) {
    qs.set("auto_benchmark", "true");
  }
  if (benchmarkGate?.minOverallScore !== undefined) {
    qs.set("min_overall_score", String(benchmarkGate.minOverallScore));
  }
  if (benchmarkGate?.minMetricScore !== undefined) {
    qs.set("min_metric_score", String(benchmarkGate.minMetricScore));
  }
  return `${BASE}/datasets/${datasetId}/export?${qs}`;
}

export function datasetImageUrl(datasetId: string, imageId: string): string {
  return `${BASE}/datasets/${datasetId}/images/${encodeURIComponent(imageId)}`;
}
