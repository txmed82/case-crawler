// ui/src/pages/PlayCasePage.tsx
import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchCase } from "../api/client";
import VignetteCard from "../components/VignetteCard";
import DecisionCards from "../components/DecisionCards";
import OutcomeReveal from "../components/OutcomeReveal";
import CaseDebrief from "../components/CaseDebrief";

type Phase = "presentation" | "decision" | "reveal" | "debrief";

export default function PlayCasePage() {
  const { caseId } = useParams();
  const [phase, setPhase] = useState<Phase>("presentation");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const { data: caseData, isLoading } = useQuery({
    queryKey: ["case", caseId],
    queryFn: () => fetchCase(caseId!),
    enabled: !!caseId,
  });

  if (isLoading) return <p>Loading case...</p>;
  if (!caseData) return <p>Case not found.</p>;

  const handleSelect = (index: number) => {
    setSelectedIndex(index);
    setPhase("reveal");
  };

  return (
    <div className="space-y-6">
      <VignetteCard caseData={caseData} />

      {phase === "presentation" && (
        <button onClick={() => setPhase("decision")}
          className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700">
          Show Choices
        </button>
      )}

      {(phase === "decision" || phase === "reveal" || phase === "debrief") && (
        <DecisionCards caseData={caseData} onSelect={handleSelect} disabled={phase !== "decision"} />
      )}

      {phase === "reveal" && selectedIndex !== null && (
        <>
          <OutcomeReveal caseData={caseData} selectedIndex={selectedIndex} />
          <button onClick={() => setPhase("debrief")}
            className="rounded-lg bg-gray-600 px-6 py-2 text-white hover:bg-gray-700">
            View Full Debrief
          </button>
        </>
      )}

      {phase === "debrief" && (
        <>
          {selectedIndex !== null && <OutcomeReveal caseData={caseData} selectedIndex={selectedIndex} />}
          <CaseDebrief caseData={caseData} />
        </>
      )}
    </div>
  );
}
