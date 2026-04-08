// ui/src/components/DecisionTreeViz.tsx
import type { GeneratedCase } from "../api/client";

export default function DecisionTreeViz({ caseData }: { caseData: GeneratedCase }) {
  return (
    <div className="space-y-2">
      <h3 className="font-semibold text-gray-700">Decision Tree</h3>
      {caseData.decision_tree.map((choice, i) => {
        const color = choice.is_correct
          ? "border-green-300 bg-green-50"
          : choice.error_type === "catastrophic"
            ? "border-red-300 bg-red-50"
            : "border-yellow-300 bg-yellow-50";
        return (
          <div key={i} className={`rounded-lg border-2 p-3 ${color}`}>
            <div className="flex items-center justify-between">
              <span className="font-medium text-sm">{choice.choice}</span>
              <span className="text-xs">
                {choice.is_correct ? "Correct" : choice.error_type}
              </span>
            </div>
            <p className="text-xs text-gray-600 mt-1">{choice.outcome}</p>
          </div>
        );
      })}
    </div>
  );
}
