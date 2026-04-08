// ui/src/components/OutcomeReveal.tsx
import type { GeneratedCase } from "../api/client";

interface Props {
  caseData: GeneratedCase;
  selectedIndex: number;
}

export default function OutcomeReveal({ caseData, selectedIndex }: Props) {
  const choice = caseData.decision_tree[selectedIndex];
  const isCorrect = choice.is_correct;

  return (
    <div className={`rounded-lg border-2 p-6 space-y-3 ${
      isCorrect ? "border-green-300 bg-green-50" : "border-red-300 bg-red-50"
    }`}>
      <div className="flex items-center gap-2">
        <span className={`text-lg font-bold ${isCorrect ? "text-green-700" : "text-red-700"}`}>
          {isCorrect ? "Correct!" : choice.error_type === "catastrophic" ? "Catastrophic Error" : "Incorrect"}
        </span>
        {choice.error_type && (
          <span className={`text-xs rounded px-2 py-0.5 ${
            choice.error_type === "catastrophic" ? "bg-red-200 text-red-800" : "bg-yellow-200 text-yellow-800"
          }`}>{choice.error_type}</span>
        )}
      </div>
      <p className="text-gray-800"><strong>Your choice:</strong> {choice.choice}</p>
      <p className="text-gray-700"><strong>Outcome:</strong> {choice.outcome}</p>
      <p className="text-gray-600"><strong>Reasoning:</strong> {choice.reasoning}</p>
      {choice.consequence && (
        <p className="text-red-700"><strong>Consequence:</strong> {choice.consequence}</p>
      )}
      {!isCorrect && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-green-700"><strong>Correct answer:</strong>{" "}
            {caseData.decision_tree.find((d) => d.is_correct)?.choice}
          </p>
        </div>
      )}
    </div>
  );
}
