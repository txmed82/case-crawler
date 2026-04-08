import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchSources, searchChunks } from '../api/client';
import ChunkCard from '../components/ChunkCard';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [selectedSources, setSelectedSources] = useState<Set<string>>(new Set());

  const { data: sourcesData } = useQuery({
    queryKey: ['sources'],
    queryFn: fetchSources,
  });

  const { data: searchData, isLoading, error } = useQuery({
    queryKey: ['search', submittedQuery, Array.from(selectedSources).sort()],
    queryFn: () =>
      searchChunks(
        submittedQuery,
        selectedSources.size > 0 ? Array.from(selectedSources) : undefined
      ),
    enabled: submittedQuery.trim().length > 0,
  });

  const availableSources = sourcesData?.sources ?? [];

  function toggleSource(id: string) {
    setSelectedSources((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    setSubmittedQuery(query);
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Search Knowledge Base</h1>
        <p className="mt-1 text-sm text-gray-500">
          Semantic search across ingested medical content.
        </p>
      </div>

      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search medical content…"
          className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={!query.trim()}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
        >
          Search
        </button>
      </form>

      {availableSources.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-600 mb-2">Filter by source:</p>
          <div className="flex flex-wrap gap-2">
            {availableSources.map((source) => (
              <button
                key={source.id}
                type="button"
                onClick={() => toggleSource(source.id)}
                className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                  selectedSources.has(source.id)
                    ? 'border-blue-400 bg-blue-100 text-blue-700'
                    : 'border-gray-300 bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                {source.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-sm text-gray-500 animate-pulse">Searching…</div>
        </div>
      )}

      {error && (
        <div className="rounded-md bg-red-50 border border-red-200 p-3 text-sm text-red-700">
          Search failed: {error instanceof Error ? error.message : 'Unknown error'}
        </div>
      )}

      {searchData && (
        <div className="space-y-3">
          <p className="text-sm text-gray-500">
            {searchData.total} result{searchData.total !== 1 ? 's' : ''} for{' '}
            <strong className="text-gray-700">&ldquo;{searchData.query}&rdquo;</strong>
          </p>
          {searchData.results.length === 0 ? (
            <div className="rounded-lg border border-dashed border-gray-300 p-8 text-center text-sm text-gray-400">
              No results found. Try ingesting content on this topic first.
            </div>
          ) : (
            <div className="space-y-3">
              {searchData.results.map((chunk) => (
                <ChunkCard key={chunk.id} chunk={chunk} />
              ))}
            </div>
          )}
        </div>
      )}

      {!submittedQuery && !isLoading && (
        <div className="rounded-lg border border-dashed border-gray-200 p-8 text-center text-sm text-gray-400">
          Enter a search query above to find relevant medical content.
        </div>
      )}
    </div>
  );
}
