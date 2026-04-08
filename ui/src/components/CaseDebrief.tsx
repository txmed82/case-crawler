// ui/src/components/CaseDebrief.tsx
import type { GeneratedCase } from "../api/client";
import DecisionTreeViz from "./DecisionTreeViz";

export default function CaseDebrief({ caseData }: { caseData: GeneratedCase }) {
  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-gray-200 bg-white p-6 space-y-3">
        <h3 className="font-semibold text-gray-700">Ground Truth</h3>
        <p><strong>Diagnosis:</strong> {caseData.ground_truth.diagnosis}</p>
        <p><strong>Optimal Next Step:</strong> {caseData.ground_truth.optimal_next_step}</p>
        <p><strong>Rationale:</strong> {caseData.ground_truth.rationale}</p>
        <div>
          <strong>Key Findings:</strong>
          <ul className="list-disc list-inside ml-2 mt-1">
            {caseData.ground_truth.key_findings.map((f, i) => (
              <li key={i} className="text-sm text-gray-600">{f}</li>
            ))}
          </ul>
        </div>
      </div>

      <DecisionTreeViz caseData={caseData} />

      {caseData.complications.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-6 space-y-3">
          <h3 className="font-semibold text-gray-700">Complications</h3>
          {caseData.complications.map((c, i) => (
            <div key={i} className="text-sm">
              <strong>{c.trigger}:</strong> {c.detail} — {c.event} ({c.outcome})
            </div>
          ))}
        </div>
      )}

      {caseData.sources.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-6 space-y-2">
          <h3 className="font-semibold text-gray-700">Sources</h3>
          {caseData.sources.map((s, i) => (
            <p key={i} className="text-sm text-gray-600">{String(s.reference || s.type || "")}</p>
          ))}
        </div>
      )}
    </div>
  );
}
