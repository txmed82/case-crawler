import { useState } from 'react';
import type { ChunkResult } from '../api/client';
import SourceBadge from './SourceBadge';

interface ChunkCardProps {
  chunk: ChunkResult;
}

export default function ChunkCard({ chunk }: ChunkCardProps) {
  const [expanded, setExpanded] = useState(false);

  const previewLength = 300;
  const isLong = chunk.content.length > previewLength;
  const displayContent = expanded || !isLong
    ? chunk.content
    : chunk.content.slice(0, previewLength) + '…';

  const scorePercent = Math.round(chunk.score * 100);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-gray-900 truncate text-sm">
            {chunk.document_url ? (
              <a
                href={chunk.document_url}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline text-blue-700"
              >
                {chunk.document_title}
              </a>
            ) : (
              chunk.document_title
            )}
          </h3>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <SourceBadge name={chunk.source_name} credibilityLevel={chunk.credibility_level} />
          <span
            className="text-xs font-mono text-gray-500"
            title={`Relevance score: ${chunk.score.toFixed(4)}`}
          >
            {scorePercent}%
          </span>
        </div>
      </div>

      <p className="text-sm text-gray-600 leading-relaxed whitespace-pre-wrap">
        {displayContent}
      </p>

      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 text-xs text-blue-600 hover:text-blue-800 font-medium"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
}
