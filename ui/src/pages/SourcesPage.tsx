import { useQuery } from '@tanstack/react-query';
import { fetchSources } from '../api/client';
import SourceBadge from '../components/SourceBadge';

const credibilityOrder = { high: 0, medium: 1, low: 2, unknown: 3 };

export default function SourcesPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['sources'],
    queryFn: fetchSources,
  });

  const available = data?.sources
    .filter((s) => s.available)
    .sort((a, b) => credibilityOrder[a.credibility_level] - credibilityOrder[b.credibility_level]) ?? [];

  const unavailable = data?.sources.filter((s) => !s.available) ?? [];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-sm text-gray-500 animate-pulse">Loading sources…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md bg-red-50 border border-red-200 p-3 text-sm text-red-700">
        Failed to load sources: {error instanceof Error ? error.message : 'Unknown error'}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Data Sources</h1>
        <p className="mt-1 text-sm text-gray-500">
          {available.length} available · {unavailable.length} require API keys
        </p>
      </div>

      <section>
        <h2 className="text-base font-semibold text-gray-800 mb-3">Available Sources</h2>
        {available.length === 0 ? (
          <p className="text-sm text-gray-400 italic">No sources available.</p>
        ) : (
          <div className="space-y-2">
            {available.map((source) => (
              <div
                key={source.id}
                className="flex items-start gap-3 rounded-lg border border-gray-200 bg-white p-4"
              >
                <div className="shrink-0 pt-0.5">
                  <SourceBadge
                    name={source.name}
                    credibilityLevel={source.credibility_level}
                    available={true}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-600">{source.description}</p>
                  <p className="text-xs text-gray-400 mt-1 font-mono">{source.id}</p>
                </div>
                <div className="shrink-0">
                  <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                    Ready
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {unavailable.length > 0 && (
        <section>
          <h2 className="text-base font-semibold text-gray-800 mb-3">Requires API Key</h2>
          <div className="space-y-2">
            {unavailable.map((source) => (
              <div
                key={source.id}
                className="flex items-start gap-3 rounded-lg border border-gray-100 bg-gray-50 p-4 opacity-70"
              >
                <div className="shrink-0 pt-0.5">
                  <SourceBadge
                    name={source.name}
                    credibilityLevel={source.credibility_level}
                    available={false}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-500">{source.description}</p>
                  <p className="text-xs text-gray-400 mt-1 font-mono">{source.id}</p>
                </div>
                <div className="shrink-0">
                  <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-500 border border-gray-300">
                    <span className="h-1.5 w-1.5 rounded-full bg-gray-400" />
                    Unavailable
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
