// ui/src/pages/CasesPage.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchCases } from "../api/client";
import { useNavigate } from "react-router-dom";

export default function CasesPage() {
  const [topicFilter, setTopicFilter] = useState("");
  const [difficultyFilter, setDifficultyFilter] = useState("");
  const navigate = useNavigate();

  const { data, isLoading } = useQuery({
    queryKey: ["cases", topicFilter, difficultyFilter],
    queryFn: () => fetchCases({
      topic: topicFilter || undefined,
      difficulty: difficultyFilter || undefined,
      limit: 50,
    }),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Cases</h1>

      <div className="flex gap-3">
        <input type="text" value={topicFilter} onChange={(e) => setTopicFilter(e.target.value)}
          placeholder="Filter by topic" className="rounded-lg border border-gray-300 px-3 py-2 text-sm" />
        <select value={difficultyFilter} onChange={(e) => setDifficultyFilter(e.target.value)}
          className="rounded-lg border border-gray-300 px-3 py-2 text-sm">
          <option value="">All difficulties</option>
          <option value="medical_student">Medical Student</option>
          <option value="resident">Resident</option>
          <option value="attending">Attending</option>
        </select>
      </div>

      {isLoading && <p className="text-sm text-gray-500">Loading...</p>}

      {data && (
        <div className="space-y-2">
          <p className="text-sm text-gray-500">{data.total} case(s)</p>
          {data.cases.map((c) => (
            <div key={c.case_id}
              onClick={() => navigate(`/play/${c.case_id}`)}
              className="flex items-center justify-between rounded-lg border border-gray-200 p-4 hover:border-blue-300 cursor-pointer">
              <div>
                <p className="font-medium">{c.topic}</p>
                <p className="text-sm text-gray-500">{c.difficulty} | {c.specialty.join(", ")}</p>
              </div>
              <div className="text-right text-sm">
                <p className="text-green-600">Acc: {(c.review?.accuracy_score ?? 0).toFixed(2)}</p>
                <p className="text-gray-400">{c.case_id.slice(0, 8)}</p>
              </div>
            </div>
          ))}
          {data.cases.length === 0 && <p className="text-gray-500">No cases yet. Generate some first.</p>}
        </div>
      )}
    </div>
  );
}
