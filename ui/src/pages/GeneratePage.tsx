import { useState } from "react";
import { startDatasetGenerate } from "../api/client";
import type { DatasetGenerateResponse, SyntheticModality } from "../api/client";

const modalityOptions: { value: SyntheticModality; label: string }[] = [
  { value: "structured_ehr", label: "EHR" },
  { value: "clinical_text", label: "Notes" },
  { value: "labs", label: "Labs" },
  { value: "vitals", label: "Vitals" },
  { value: "time_series", label: "Time series" },
  { value: "imaging", label: "Imaging" },
];

const sexOptions = ["female", "male", "other"] as const;
type SexOption = (typeof sexOptions)[number];

export default function GeneratePage() {
  const [topic, setTopic] = useState("");
  const [complexity, setComplexity] = useState<"simple" | "moderate" | "complex" | "rare">("moderate");
  const [count, setCount] = useState(1);
  const [ageMin, setAgeMin] = useState("");
  const [ageMax, setAgeMax] = useState("");
  const [sexes, setSexes] = useState<SexOption[]>([]);
  const [baseTime, setBaseTime] = useState("");
  const [modalities, setModalities] = useState<SyntheticModality[]>([
    "structured_ehr",
    "clinical_text",
    "labs",
    "vitals",
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<DatasetGenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!topic.trim() || modalities.length === 0 || isGenerating) return;
    if (!Number.isInteger(count) || count < 1) {
      setError("Record count must be a positive integer.");
      return;
    }
    const parsedAgeMin = ageMin === "" ? undefined : Number(ageMin);
    const parsedAgeMax = ageMax === "" ? undefined : Number(ageMax);
    if (
      (parsedAgeMin !== undefined && (!Number.isInteger(parsedAgeMin) || parsedAgeMin < 0)) ||
      (parsedAgeMax !== undefined && (!Number.isInteger(parsedAgeMax) || parsedAgeMax < 0))
    ) {
      setError("Age limits must be non-negative whole numbers.");
      return;
    }
    if (
      parsedAgeMin !== undefined &&
      parsedAgeMax !== undefined &&
      parsedAgeMin > parsedAgeMax
    ) {
      setError("Minimum age cannot be greater than maximum age.");
      return;
    }
    const cohortConstraints: Record<string, unknown> = {};
    if (parsedAgeMin !== undefined) cohortConstraints.age_min = parsedAgeMin;
    if (parsedAgeMax !== undefined) cohortConstraints.age_max = parsedAgeMax;
    if (sexes.length > 0) cohortConstraints.sexes = sexes;
    if (baseTime) cohortConstraints.base_time = baseTime;

    setResult(null);
    setError(null);
    setIsGenerating(true);
    try {
      const resp = await startDatasetGenerate({
        topic: topic.trim(),
        complexity,
        count,
        modalities,
        ...(Object.keys(cohortConstraints).length > 0
          ? { cohort_constraints: cohortConstraints }
          : {}),
      });
      setResult(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  const toggleModality = (modality: SyntheticModality) => {
    setModalities((current) =>
      current.includes(modality)
        ? current.filter((item) => item !== modality)
        : [...current, modality]
    );
  };

  const toggleSex = (sex: SexOption) => {
    setSexes((current) =>
      current.includes(sex) ? current.filter((item) => item !== sex) : [...current, sex]
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Generate Dataset</h1>
        <p className="mt-1 text-sm text-gray-600">
          Create synthetic healthcare records with structured EHR fields, notes, labs, vitals,
          time series, and image placeholders.
        </p>
      </div>

      <div className="space-y-4">
        <input
          id="dataset-topic"
          aria-label="Dataset topic"
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. sepsis, heart failure exacerbation, diabetic ketoacidosis"
          className="w-full rounded-lg border border-gray-300 px-4 py-2"
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
        />
        <div className="flex flex-wrap gap-2">
          {modalityOptions.map((option) => {
            const selected = modalities.includes(option.value);
            return (
              <button
                key={option.value}
                type="button"
                onClick={() => toggleModality(option.value)}
                aria-pressed={selected}
                className={`rounded-md border px-3 py-2 text-sm ${
                  selected
                    ? "border-blue-600 bg-blue-50 text-blue-700"
                    : "border-gray-300 text-gray-600 hover:bg-gray-50"
                }`}
              >
                {option.label}
              </button>
            );
          })}
        </div>
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
        </div>

        <div className="grid gap-4 md:grid-cols-[repeat(2,minmax(0,14rem))_minmax(0,18rem)]">
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Minimum age</span>
            <input
              id="dataset-age-min"
              aria-label="Minimum age"
              type="number"
              value={ageMin}
              onChange={(e) => setAgeMin(e.target.value)}
              min={0}
              max={120}
              placeholder="Any"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Maximum age</span>
            <input
              id="dataset-age-max"
              aria-label="Maximum age"
              type="number"
              value={ageMax}
              onChange={(e) => setAgeMax(e.target.value)}
              min={0}
              max={120}
              placeholder="Any"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Base time</span>
            <input
              id="dataset-base-time"
              aria-label="Base time"
              type="datetime-local"
              value={baseTime}
              onChange={(e) => setBaseTime(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-sm font-medium text-gray-700">Sex mix</span>
          {sexOptions.map((sex) => {
            const selected = sexes.includes(sex);
            return (
              <button
                key={sex}
                type="button"
                onClick={() => toggleSex(sex)}
                aria-pressed={selected}
                className={`rounded-md border px-3 py-2 text-sm capitalize ${
                  selected
                    ? "border-blue-600 bg-blue-50 text-blue-700"
                    : "border-gray-300 text-gray-600 hover:bg-gray-50"
                }`}
              >
                {sex}
              </button>
            );
          })}
        </div>

        <div>
          <button
            onClick={handleGenerate}
            disabled={
              !topic.trim() ||
              modalities.length === 0 ||
              !Number.isInteger(count) ||
              count < 1 ||
              isGenerating
            }
            className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Generate
          </button>
        </div>
      </div>

      {isGenerating && <div className="text-sm text-gray-600">Generating synthetic records...</div>}

      {result && (
        <div className="rounded-lg border border-green-200 bg-green-50 p-4">
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
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="font-medium text-red-800">Generation failed</p>
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}
    </div>
  );
}
