// ui/src/pages/GeneratePage.tsx
import { useState } from "react";
import { startGenerate, getGenerateStatus } from "../api/client";
import type { GenerateJobResponse } from "../api/client";
import { useQuery } from "@tanstack/react-query";

export default function GeneratePage() {
  const [topic, setTopic] = useState("");
  const [difficulty, setDifficulty] = useState("resident");
  const [count, setCount] = useState(1);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateJobResponse | null>(null);

  const { data: jobStatus } = useQuery({
    queryKey: ["generate-status", jobId],
    queryFn: () => getGenerateStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => query.state.data?.status === "running" ? 2000 : false,
  });

  if (jobStatus && jobStatus.status !== "running" && jobId) {
    if (!result || result.job_id !== jobStatus.job_id) {
      setResult(jobStatus);
      setJobId(null);
    }
  }

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    setResult(null);
    const resp = await startGenerate({ topic: topic.trim(), difficulty, count });
    setJobId(resp.job_id);
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Generate Cases</h1>

      <div className="space-y-4">
        <input
          type="text" value={topic} onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. subarachnoid hemorrhage"
          className="w-full rounded-lg border border-gray-300 px-4 py-2"
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
        />
        <div className="flex gap-4">
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}
            className="rounded-lg border border-gray-300 px-3 py-2">
            <option value="medical_student">Medical Student</option>
            <option value="resident">Resident</option>
            <option value="attending">Attending</option>
          </select>
          <input type="number" value={count} onChange={(e) => setCount(Number(e.target.value))}
            min={1} max={100} className="w-24 rounded-lg border border-gray-300 px-3 py-2" />
          <button onClick={handleGenerate} disabled={!topic.trim() || !!jobId}
            className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700 disabled:opacity-50">
            Generate
          </button>
        </div>
      </div>

      {jobId && <div className="text-sm text-gray-600">Generating cases... (this may take a minute)</div>}

      {result && result.status === "completed" && (
        <div className="rounded-lg bg-green-50 border border-green-200 p-4">
          <p className="font-medium text-green-800">Generation complete ({result.elapsed_seconds}s)</p>
          <p className="text-sm text-green-700">{result.cases_generated} generated, {result.cases_failed} failed</p>
        </div>
      )}

      {result && result.status === "failed" && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4">
          <p className="font-medium text-red-800">Generation failed</p>
          <p className="text-sm text-red-700">{result.error}</p>
        </div>
      )}
    </div>
  );
}
