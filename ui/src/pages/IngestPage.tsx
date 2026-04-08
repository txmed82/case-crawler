import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchSources, startIngest } from '../api/client';
import type { IngestStatusResponse } from '../api/client';
import JobProgress from '../components/JobProgress';
import SourceBadge from '../components/SourceBadge';

export default function IngestPage() {
  const [topic, setTopic] = useState('');
  const [selectedSources, setSelectedSources] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string | null>(null);
  const [completedJob, setCompletedJob] = useState<IngestStatusResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: sourcesData } = useQuery({
    queryKey: ['sources'],
    queryFn: fetchSources,
  });

  const availableSources = sourcesData?.sources.filter((s) => s.available) ?? [];

  function toggleSource(id: string) {
    setSelectedSources((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleAll() {
    if (selectedSources.size === availableSources.length) {
      setSelectedSources(new Set());
    } else {
      setSelectedSources(new Set(availableSources.map((s) => s.id)));
    }
  }

  async function handleIngest(e: React.FormEvent) {
    e.preventDefault();
    if (!topic.trim()) return;
    setError(null);
    setCompletedJob(null);
    setJobId(null);
    setIsSubmitting(true);
    try {
      const response = await startIngest({
        topic: topic.trim(),
        source_ids: selectedSources.size > 0 ? Array.from(selectedSources) : undefined,
      });
      setJobId(response.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start ingest');
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleJobComplete(result: IngestStatusResponse) {
    setCompletedJob(result);
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Ingest Medical Content</h1>
        <p className="mt-1 text-sm text-gray-500">
          Enter a topic to crawl and ingest content from medical data sources into the knowledge base.
        </p>
      </div>

      <form onSubmit={handleIngest} className="space-y-5">
        <div>
          <label htmlFor="topic" className="block text-sm font-medium text-gray-700 mb-1">
            Topic
          </label>
          <input
            id="topic"
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g. septic shock management, beta-blocker overdose"
            className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium text-gray-700">
              Sources ({availableSources.length} available)
            </label>
            {availableSources.length > 0 && (
              <button
                type="button"
                onClick={toggleAll}
                className="text-xs text-blue-600 hover:text-blue-800"
              >
                {selectedSources.size === availableSources.length ? 'Deselect all' : 'Select all'}
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {availableSources.map((source) => (
              <label
                key={source.id}
                className={`flex items-center gap-2.5 rounded-md border px-3 py-2 cursor-pointer transition-colors ${
                  selectedSources.has(source.id)
                    ? 'border-blue-400 bg-blue-50'
                    : 'border-gray-200 bg-white hover:bg-gray-50'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedSources.has(source.id)}
                  onChange={() => toggleSource(source.id)}
                  className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <div className="flex-1 min-w-0">
                  <SourceBadge name={source.name} credibilityLevel={source.credibility_level} />
                </div>
              </label>
            ))}
          </div>

          {availableSources.length === 0 && (
            <p className="text-sm text-gray-400 italic">Loading sources…</p>
          )}
          {selectedSources.size === 0 && availableSources.length > 0 && (
            <p className="text-xs text-gray-500 mt-1">No sources selected — all available sources will be used.</p>
          )}
        </div>

        <button
          type="submit"
          disabled={!topic.trim() || isSubmitting}
          className="w-full rounded-md bg-blue-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
        >
          {isSubmitting ? 'Starting…' : 'Start Ingest'}
        </button>
      </form>

      {error && (
        <div className="rounded-md bg-red-50 border border-red-200 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {jobId && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">Ingest Progress</h2>
          <JobProgress jobId={jobId} onComplete={handleJobComplete} />
        </div>
      )}

      {completedJob && completedJob.status === 'completed' && (
        <div className="rounded-lg border border-green-200 bg-green-50 p-4">
          <h2 className="text-sm font-semibold text-green-800 mb-1">Ingest Complete</h2>
          <p className="text-sm text-green-700">
            Successfully stored <strong>{completedJob.chunks_stored}</strong> chunks for topic:{' '}
            <em>{completedJob.topic}</em>
          </p>
          {completedJob.completed_at && (
            <p className="text-xs text-green-600 mt-1">
              Completed at {new Date(completedJob.completed_at).toLocaleTimeString()}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
