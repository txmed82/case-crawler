import { useQuery } from '@tanstack/react-query';
import { getIngestStatus } from '../api/client';
import type { IngestStatusResponse } from '../api/client';

interface JobProgressProps {
  jobId: string;
  onComplete?: (result: IngestStatusResponse) => void;
}

const statusColors = {
  pending: 'bg-gray-400',
  running: 'bg-blue-500',
  completed: 'bg-green-500',
  failed: 'bg-red-500',
};

const statusLabels = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
};

export default function JobProgress({ jobId, onComplete }: JobProgressProps) {
  const { data, error } = useQuery({
    queryKey: ['ingest-status', jobId],
    queryFn: () => getIngestStatus(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'completed' || status === 'failed') {
        if (onComplete && query.state.data) onComplete(query.state.data);
        return false;
      }
      return 1500;
    },
  });

  if (error) {
    return (
      <div className="rounded-md bg-red-50 border border-red-200 p-3 text-sm text-red-700">
        Error polling job status: {error instanceof Error ? error.message : 'Unknown error'}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-sm text-gray-500 animate-pulse">Loading job status…</div>
    );
  }

  const progressPct =
    data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-gray-700">
          Job{' '}
          <span className="font-mono text-xs text-gray-500">{jobId.slice(0, 8)}…</span>
        </span>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium text-white ${statusColors[data.status]}`}
        >
          {statusLabels[data.status]}
        </span>
      </div>

      <div className="h-2 w-full rounded-full bg-gray-200 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${statusColors[data.status]}`}
          style={{ width: `${data.status === 'completed' ? 100 : progressPct}%` }}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500">
        <span>
          {data.progress} / {data.total} sources
        </span>
        <span>{data.chunks_stored} chunks stored</span>
      </div>

      {data.errors.length > 0 && (
        <div className="rounded-md bg-yellow-50 border border-yellow-200 p-2 text-xs text-yellow-800">
          <strong>Warnings:</strong>
          <ul className="mt-1 list-disc list-inside space-y-0.5">
            {data.errors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
