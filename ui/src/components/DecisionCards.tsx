// ui/src/components/DecisionCards.tsx
import type { GeneratedCase } from "../api/client";

interface Props {
  caseData: GeneratedCase;
  onSelect: (index: number) => void;
  disabled: boolean;
}

export default function DecisionCards({ caseData, onSelect, disabled }: Props) {
  return (
    <div className="space-y-3">
      {caseData.decision_tree.map((choice, i) => (
        <button key={i} onClick={() => onSelect(i)} disabled={disabled}
          className="w-full text-left rounded-lg border border-gray-200 p-4 hover:border-blue-400 hover:bg-blue-50 transition-colors disabled:opacity-60 disabled:hover:border-gray-200 disabled:hover:bg-white">
          <p className="font-medium text-gray-800">{choice.choice}</p>
        </button>
      ))}
    </div>
  );
}
