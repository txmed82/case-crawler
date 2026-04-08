// ui/src/components/VignetteCard.tsx
import type { GeneratedCase } from "../api/client";

export default function VignetteCard({ caseData }: { caseData: GeneratedCase }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 space-y-4">
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <span className="rounded bg-blue-100 px-2 py-0.5 text-blue-700">{caseData.difficulty}</span>
        {caseData.specialty.map((s) => (
          <span key={s} className="rounded bg-gray-100 px-2 py-0.5">{s}</span>
        ))}
      </div>
      <div className="text-sm text-gray-500">
        Patient: {caseData.patient.age}yo {caseData.patient.sex} | {caseData.patient.demographics}
      </div>
      <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">{caseData.vignette}</p>
      <p className="text-lg font-semibold text-gray-900">{caseData.decision_prompt}</p>
    </div>
  );
}
