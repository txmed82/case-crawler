import { useState } from "react";
import { startDatasetGenerate } from "../api/client";
import type { DatasetGenerateResponse } from "../api/client";

export default function GeneratePage() {
  const [topic, setTopic] = useState("");
  const [complexity, setComplexity] = useState<"simple" | "moderate" | "complex" | "rare">("moderate");
  const [count, setCount] = useState(1);
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<DatasetGenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!topic.trim() || isGenerating) return;
    setResult(null);
    setError(null);
    setIsGenerating(true);
    try {
      const resp = await startDatasetGenerate({ topic: topic.trim(), complexity, count });
      setResult(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Generate Dataset</h1>

      <div className="space-y-4">
        <input
          id="dataset-topic"
          aria-label="Dataset topic"
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. sepsis"
          className="w-full rounded-lg border border-gray-300 px-4 py-2"
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
        />
        <div className="flex flex-wrap gap-4">
          <select
            id="dataset-complexity"
            aria-label="Complexity"
            value={complexity}
            onChange={(e) => setComplexity(e.target.value as typeof complexity)}
            className="rounded-lg border border-gray-300 px-3 py-2"
          >
            <option value="simple">Simple</option>
            <option value="moderate">Moderate</option>
            <option value="complex">Complex</option>
            <option value="rare">Rare</option>
          </select>
          <input
            id="dataset-count"
            aria-label="Record count"
            type="number"
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
            min={1}
            max={100}
            className="w-24 rounded-lg border border-gray-300 px-3 py-2"
          />
          <button
            onClick={handleGenerate}
            disabled={!topic.trim() || isGenerating}
            className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Generate
          </button>
        </div>
      </div>

      {isGenerating && <div className="text-sm text-gray-600">Generating synthetic records...</div>}

      {result && (
        <div className="rounded-lg bg-green-50 border border-green-200 p-4">
          <p className="font-medium text-green-800">Dataset generated</p>
          <p className="text-sm text-green-700">
            {result.generated} generated, {result.approved} approved
          </p>
          <p className="text-sm text-green-700">
            Showing {result.records.length} of {result.total_records} records
          </p>
          <p className="text-xs text-green-700">{result.dataset_id}</p>
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4">
          <p className="font-medium text-red-800">Generation failed</p>
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}
    </div>
  );
}
