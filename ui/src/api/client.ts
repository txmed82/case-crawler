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
