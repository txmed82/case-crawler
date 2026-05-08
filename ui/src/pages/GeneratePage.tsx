import { useEffect, useState } from "react";
import {
  fetchDatasetCapabilities,
  fetchReferenceDatasetCatalog,
  generateReleasePackage,
  importReferenceDataset,
  importSyntheaFhir,
  startDatasetGenerate,
} from "../api/client";
import type {
  DatasetCapabilitiesResponse,
  DatasetGenerateResponse,
  ExportFormat,
  ReferenceDatasetCatalogItem,
  ReferenceDatasetImportResponse,
  ReleasePackageResponse,
  SyntheaImportResponse,
  SyntheticModality,
} from "../api/client";

const modalityOptions: { value: SyntheticModality; label: string }[] = [
  { value: "structured_ehr", label: "EHR" },
  { value: "clinical_text", label: "Notes" },
  { value: "labs", label: "Labs" },
  { value: "vitals", label: "Vitals" },
  { value: "time_series", label: "Time series" },
  { value: "imaging", label: "Imaging" },
];

const exportFormatOptions: { value: ExportFormat; label: string }[] = [
  { value: "sft_jsonl", label: "SFT" },
  { value: "note_fact_sft_jsonl", label: "Note facts" },
  { value: "clinical_observation_jsonl", label: "Observations" },
  { value: "medication_reconciliation_jsonl", label: "Medications" },
  { value: "chat_jsonl", label: "Chat" },
  { value: "tool_call_jsonl", label: "Tool calls" },
  { value: "multimodal_jsonl", label: "Multimodal" },
  { value: "time_series_jsonl", label: "Time series" },
  { value: "dpo_jsonl", label: "DPO" },
  { value: "rl_jsonl", label: "RL" },
  { value: "fhir_ndjson", label: "FHIR" },
  { value: "parquet", label: "Parquet" },
  { value: "raw_jsonl", label: "Raw" },
];

const llmProviderOptions = ["anthropic", "openai", "openrouter", "ollama"] as const;
const clinicalTextNoiseOptions = [
  { value: "standard", label: "Standard" },
  { value: "message", label: "Message" },
  { value: "ocr", label: "OCR" },
  { value: "heavy", label: "Heavy" },
] as const;

const sexOptions = ["female", "male", "other"] as const;
type SexOption = (typeof sexOptions)[number];
type LlmProviderOption = (typeof llmProviderOptions)[number];
type ClinicalTextNoiseProfile = (typeof clinicalTextNoiseOptions)[number]["value"];
type ReferenceImportMode = "registered" | "custom" | "local";

export default function GeneratePage() {
  const [topic, setTopic] = useState("");
  const [recipe, setRecipe] = useState("");
  const [complexity, setComplexity] = useState<"simple" | "moderate" | "complex" | "rare">("moderate");
  const [count, setCount] = useState(1);
  const [ageMin, setAgeMin] = useState("");
  const [ageMax, setAgeMax] = useState("");
  const [sexes, setSexes] = useState<SexOption[]>([]);
  const [topicMix, setTopicMix] = useState("");
  const [baseTime, setBaseTime] = useState("");
  const [encounterCount, setEncounterCount] = useState("");
  const [races, setRaces] = useState("");
  const [ethnicities, setEthnicities] = useState("");
  const [insurance, setInsurance] = useState("");
  const [smokingStatuses, setSmokingStatuses] = useState("");
  const [alcoholUse, setAlcoholUse] = useState("");
  const [housing, setHousing] = useState("");
  const [modalities, setModalities] = useState<SyntheticModality[]>([
    "structured_ehr",
    "clinical_text",
    "labs",
    "vitals",
  ]);
  const [exportFormats, setExportFormats] = useState<ExportFormat[]>(["sft_jsonl"]);
  const [clinicalTextBackend, setClinicalTextBackend] =
    useState<"deterministic" | "llm" | "external">("deterministic");
  const [clinicalTextNoiseProfile, setClinicalTextNoiseProfile] =
    useState<ClinicalTextNoiseProfile>("standard");
  const [clinicalTextProfile, setClinicalTextProfile] = useState("");
  const [clinicalTextCommand, setClinicalTextCommand] = useState("");
  const [llmProvider, setLlmProvider] = useState<LlmProviderOption>("ollama");
  const [llmModel, setLlmModel] = useState("");
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState("");
  const [imagingBackend, setImagingBackend] =
    useState<"placeholder" | "diffusers" | "external">("placeholder");
  const [imagingProfile, setImagingProfile] = useState("");
  const [diffusersModelId, setDiffusersModelId] = useState("");
  const [imagingCommand, setImagingCommand] = useState("");
  const [timeSeriesBackend, setTimeSeriesBackend] =
    useState<"deterministic" | "external">("deterministic");
  const [timeSeriesProfile, setTimeSeriesProfile] = useState("");
  const [timeSeriesCommand, setTimeSeriesCommand] = useState("");
  const [capabilities, setCapabilities] = useState<DatasetCapabilitiesResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<DatasetGenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isGeneratingRelease, setIsGeneratingRelease] = useState(false);
  const [releaseResult, setReleaseResult] = useState<ReleasePackageResponse | null>(null);
  const [releaseError, setReleaseError] = useState<string | null>(null);
  const [releaseFixtureLimit, setReleaseFixtureLimit] = useState("1");
  const [releaseMinOverallScore, setReleaseMinOverallScore] = useState("0.1");
  const [releaseMinMetricScore, setReleaseMinMetricScore] = useState("0");
  const [releaseTrainRatio, setReleaseTrainRatio] = useState("0.8");
  const [releaseValidationRatio, setReleaseValidationRatio] = useState("0.1");
  const [releaseTestRatio, setReleaseTestRatio] = useState("0.1");
  const [referenceCatalog, setReferenceCatalog] = useState<ReferenceDatasetCatalogItem[]>([]);
  const [referenceImportMode, setReferenceImportMode] =
    useState<ReferenceImportMode>("registered");
  const [referenceKey, setReferenceKey] = useState("");
  const [referenceDatasetId, setReferenceDatasetId] = useState("");
  const [referenceRepoId, setReferenceRepoId] = useState("");
  const [referenceLocalPath, setReferenceLocalPath] = useState("");
  const [referenceSplit, setReferenceSplit] = useState("");
  const [referenceLicense, setReferenceLicense] = useState("");
  const [referenceNoteField, setReferenceNoteField] = useState("note");
  const [referenceQuestionField, setReferenceQuestionField] = useState("");
  const [referenceAnswerField, setReferenceAnswerField] = useState("");
  const [referenceTaskField, setReferenceTaskField] = useState("");
  const [referencePatientIdField, setReferencePatientIdField] = useState("");
  const [referenceImageField, setReferenceImageField] = useState("");
  const [referenceImageLabelField, setReferenceImageLabelField] = useState("");
  const [referenceLabValuesField, setReferenceLabValuesField] = useState("");
  const [referenceVitalValuesField, setReferenceVitalValuesField] = useState("");
  const [referenceMedicationsField, setReferenceMedicationsField] = useState("");
  const [referenceTimeSeriesField, setReferenceTimeSeriesField] = useState("");
  const [referenceLimit, setReferenceLimit] = useState("25");
  const [isImportingReference, setIsImportingReference] = useState(false);
  const [referenceImportResult, setReferenceImportResult] =
    useState<ReferenceDatasetImportResponse | null>(null);
  const [referenceError, setReferenceError] = useState<string | null>(null);
  const [syntheaPath, setSyntheaPath] = useState("");
  const [syntheaDatasetId, setSyntheaDatasetId] = useState("");
  const [isImportingSynthea, setIsImportingSynthea] = useState(false);
  const [syntheaImportResult, setSyntheaImportResult] =
    useState<SyntheaImportResponse | null>(null);
  const [syntheaError, setSyntheaError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchReferenceDatasetCatalog()
      .then((resp) => {
        if (!active) return;
        setReferenceCatalog(resp.datasets);
        setReferenceKey((current) => current || resp.datasets[0]?.key || "");
      })
      .catch((err) => {
        if (!active) return;
        setReferenceError(
          err instanceof Error ? err.message : "Failed to load reference datasets"
        );
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    fetchDatasetCapabilities()
      .then((resp) => {
        if (!active) return;
        setCapabilities(resp);
      })
      .catch(() => {
        if (!active) return;
        setCapabilities(null);
      });
    return () => {
      active = false;
    };
  }, []);

  const buildCohortConstraints = (): { constraints: Record<string, unknown>; error?: string } => {
    const parsedAgeMin = ageMin === "" ? undefined : Number(ageMin);
    const parsedAgeMax = ageMax === "" ? undefined : Number(ageMax);
    const parsedEncounterCount = encounterCount === "" ? undefined : Number(encounterCount);
    if (
      (parsedAgeMin !== undefined && (!Number.isInteger(parsedAgeMin) || parsedAgeMin < 0)) ||
      (parsedAgeMax !== undefined && (!Number.isInteger(parsedAgeMax) || parsedAgeMax < 0))
    ) {
      return { constraints: {}, error: "Age limits must be non-negative whole numbers." };
    }
    if (
      parsedEncounterCount !== undefined &&
      (!Number.isInteger(parsedEncounterCount) ||
        parsedEncounterCount < 1 ||
        parsedEncounterCount > 30)
    ) {
      return { constraints: {}, error: "Encounter count must be a whole number from 1 to 30." };
    }
    if (
      parsedAgeMin !== undefined &&
      parsedAgeMax !== undefined &&
      parsedAgeMin > parsedAgeMax
    ) {
      return { constraints: {}, error: "Minimum age cannot be greater than maximum age." };
    }
    const constraints: Record<string, unknown> = {};
    const splitValues = (value: string) =>
      value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
    if (parsedAgeMin !== undefined) constraints.age_min = parsedAgeMin;
    if (parsedAgeMax !== undefined) constraints.age_max = parsedAgeMax;
    if (sexes.length > 0) constraints.sexes = sexes;
    const parsedTopicMix = splitValues(topicMix);
    const parsedRaces = splitValues(races);
    const parsedEthnicities = splitValues(ethnicities);
    const parsedInsurance = splitValues(insurance);
    const parsedSmokingStatuses = splitValues(smokingStatuses);
    const parsedAlcoholUse = splitValues(alcoholUse);
    const parsedHousing = splitValues(housing);
    if (parsedTopicMix.length > 0) constraints.topic_mix = parsedTopicMix;
    if (parsedRaces.length > 0) constraints.races = parsedRaces;
    if (parsedEthnicities.length > 0) constraints.ethnicities = parsedEthnicities;
    if (parsedInsurance.length > 0) constraints.insurance = parsedInsurance;
    if (parsedSmokingStatuses.length > 0) {
      constraints.smoking_statuses = parsedSmokingStatuses;
    }
    if (parsedAlcoholUse.length > 0) constraints.alcohol_use = parsedAlcoholUse;
    if (parsedHousing.length > 0) constraints.housing = parsedHousing;
    if (baseTime) constraints.base_time = baseTime;
    if (parsedEncounterCount !== undefined) {
      constraints.encounter_count = parsedEncounterCount;
    }
    return { constraints };
  };

  const handleGenerate = async () => {
    if (!topic.trim() || modalities.length === 0 || exportFormats.length === 0 || isGenerating) {
      return;
    }
    if (!Number.isInteger(count) || count < 1) {
      setError("Record count must be a positive integer.");
      return;
    }
    const cohort = buildCohortConstraints();
    if (cohort.error) {
      setError(cohort.error);
      return;
    }
    const cohortConstraints = cohort.constraints;
    const includesImaging = modalities.includes("imaging");
    const includesTimeSeries = modalities.includes("time_series");
    const includesClinicalText = modalities.includes("clinical_text");
    const parsedClinicalTextCommand = clinicalTextCommand
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    const parsedImagingCommand = imagingCommand
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    const parsedTimeSeriesCommand = timeSeriesCommand
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);

    setResult(null);
    setError(null);
    setIsGenerating(true);
    try {
      const resp = await startDatasetGenerate({
        topic: topic.trim(),
        ...(recipe ? { recipe } : {}),
        complexity,
        count,
        modalities,
        export_formats: exportFormats,
        ...(includesClinicalText ? { clinical_text_backend: clinicalTextBackend } : {}),
        ...(includesClinicalText
          ? { clinical_text_noise_profile: clinicalTextNoiseProfile }
          : {}),
        ...(includesClinicalText && clinicalTextBackend === "llm"
          ? { llm_provider: llmProvider }
          : {}),
        ...(includesClinicalText && clinicalTextBackend === "external" && clinicalTextProfile
          ? { clinical_text_model_profile: clinicalTextProfile }
          : {}),
        ...(includesClinicalText &&
        clinicalTextBackend === "external" &&
        parsedClinicalTextCommand.length > 0
          ? { clinical_text_command: parsedClinicalTextCommand }
          : {}),
        ...(includesClinicalText && clinicalTextBackend === "llm" && llmModel.trim()
          ? { llm_model: llmModel.trim() }
          : {}),
        ...(includesClinicalText &&
        clinicalTextBackend === "llm" &&
        ollamaBaseUrl.trim()
          ? { ollama_base_url: ollamaBaseUrl.trim() }
          : {}),
        ...(includesImaging ? { imaging_backend: imagingBackend } : {}),
        ...(includesImaging && imagingBackend === "diffusers" && imagingProfile
          ? { imaging_model_profile: imagingProfile }
          : {}),
        ...(includesImaging && imagingBackend === "diffusers" && diffusersModelId.trim()
          ? { diffusers_model_id: diffusersModelId.trim() }
          : {}),
        ...(includesImaging && imagingBackend === "external" && parsedImagingCommand.length > 0
          ? { imaging_command: parsedImagingCommand }
          : {}),
        ...(includesTimeSeries ? { time_series_backend: timeSeriesBackend } : {}),
        ...(includesTimeSeries && timeSeriesProfile
          ? { time_series_model_profile: timeSeriesProfile }
          : {}),
        ...(includesTimeSeries && parsedTimeSeriesCommand.length > 0
          ? { time_series_command: parsedTimeSeriesCommand }
          : {}),
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

  const handleGenerateReleasePackage = async () => {
    if (!topic.trim() || isGeneratingRelease) return;
    if (modalities.length === 0) {
      setReleaseError("Select at least one modality.");
      return;
    }
    if (!Number.isInteger(count) || count < 1) {
      setReleaseError("Record count must be a positive integer.");
      return;
    }
    const parsedFixtureLimit = Number(releaseFixtureLimit);
    const parsedMinOverallScore = Number(releaseMinOverallScore);
    const parsedMinMetricScore = Number(releaseMinMetricScore);
    const parsedTrainRatio = Number(releaseTrainRatio);
    const parsedValidationRatio = Number(releaseValidationRatio);
    const parsedTestRatio = Number(releaseTestRatio);
    if (!Number.isInteger(parsedFixtureLimit) || parsedFixtureLimit < 1) {
      setReleaseError("Fixture limit must be a positive integer.");
      return;
    }
    if (
      Number.isNaN(parsedMinOverallScore) ||
      parsedMinOverallScore < 0 ||
      parsedMinOverallScore > 1 ||
      Number.isNaN(parsedMinMetricScore) ||
      parsedMinMetricScore < 0 ||
      parsedMinMetricScore > 1
    ) {
      setReleaseError("Benchmark thresholds must be numbers from 0 to 1.");
      return;
    }
    if (
      Number.isNaN(parsedTrainRatio) ||
      parsedTrainRatio < 0 ||
      Number.isNaN(parsedValidationRatio) ||
      parsedValidationRatio < 0 ||
      Number.isNaN(parsedTestRatio) ||
      parsedTestRatio < 0 ||
      parsedTrainRatio + parsedValidationRatio + parsedTestRatio <= 0
    ) {
      setReleaseError("Split ratios must be non-negative and sum above zero.");
      return;
    }
    setReleaseResult(null);
    setReleaseError(null);
    setIsGeneratingRelease(true);
    try {
      const cohort = buildCohortConstraints();
      if (cohort.error) {
        setReleaseError(cohort.error);
        return;
      }
      const cohortConstraints = cohort.constraints;
      const parsedTimeSeriesCommand = timeSeriesCommand
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
      const resp = await generateReleasePackage({
        topic: topic.trim(),
        count,
        recipe: recipe || "full_multimodal_acute_care",
        complexity,
        modalities,
        export_format: "multimodal_jsonl",
        seed: "casecrawler",
        train_ratio: parsedTrainRatio,
        validation_ratio: parsedValidationRatio,
        test_ratio: parsedTestRatio,
        fixture_limit: parsedFixtureLimit,
        min_overall_score: parsedMinOverallScore,
        min_metric_score: parsedMinMetricScore,
        ...(Object.keys(cohortConstraints).length > 0
          ? { cohort_constraints: cohortConstraints }
          : {}),
        clinical_text_backend: clinicalTextBackend,
        clinical_text_noise_profile: clinicalTextNoiseProfile,
        ...(clinicalTextBackend === "llm" ? { llm_provider: llmProvider } : {}),
        ...(clinicalTextBackend === "llm" && llmModel.trim()
          ? { llm_model: llmModel.trim() }
          : {}),
        ...(clinicalTextBackend === "llm" && ollamaBaseUrl.trim()
          ? { ollama_base_url: ollamaBaseUrl.trim() }
          : {}),
        ...(clinicalTextBackend === "external" && clinicalTextProfile
          ? { clinical_text_model_profile: clinicalTextProfile }
          : {}),
        ...(clinicalTextBackend === "external" && clinicalTextCommand.trim()
          ? {
              clinical_text_command: clinicalTextCommand
                .split(",")
                .map((value) => value.trim())
                .filter(Boolean),
            }
          : {}),
        imaging_backend: imagingBackend,
        ...(imagingBackend === "diffusers" && imagingProfile
          ? { imaging_model_profile: imagingProfile }
          : {}),
        ...(imagingBackend === "diffusers" && diffusersModelId.trim()
          ? { diffusers_model_id: diffusersModelId.trim() }
          : {}),
        ...(imagingBackend === "external" && imagingCommand.trim()
          ? {
              imaging_command: imagingCommand
                .split(",")
                .map((value) => value.trim())
                .filter(Boolean),
            }
          : {}),
        time_series_backend: timeSeriesBackend,
        ...(timeSeriesBackend === "external" && timeSeriesProfile
          ? { time_series_model_profile: timeSeriesProfile }
          : {}),
        ...(timeSeriesBackend === "external" && parsedTimeSeriesCommand.length > 0
          ? { time_series_command: parsedTimeSeriesCommand }
          : {}),
      });
      setReleaseResult(resp);
      downloadBlob(resp.blob, resp.filename);
    } catch (err) {
      setReleaseError(err instanceof Error ? err.message : "Release package generation failed");
    } finally {
      setIsGeneratingRelease(false);
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

  const toggleExportFormat = (format: ExportFormat) => {
    setExportFormats((current) =>
      current.includes(format)
        ? current.filter((item) => item !== format)
        : [...current, format]
    );
  };

  const handleReferenceImport = async () => {
    if (
      !referenceDatasetId.trim() ||
      isImportingReference ||
      (referenceImportMode === "registered" && !referenceKey) ||
      (referenceImportMode === "custom" && !referenceRepoId.trim()) ||
      (referenceImportMode === "local" && !referenceLocalPath.trim())
    ) {
      return;
    }
    if (referenceImportMode !== "registered" && !referenceNoteField.trim()) {
      setReferenceError("Mapped reference imports require a note field.");
      return;
    }
    const parsedLimit = referenceLimit === "" ? undefined : Number(referenceLimit);
    if (
      parsedLimit !== undefined &&
      (!Number.isInteger(parsedLimit) || parsedLimit < 1)
    ) {
      setReferenceError("Reference import limit must be a positive integer.");
      return;
    }
    setReferenceImportResult(null);
    setReferenceError(null);
    setIsImportingReference(true);
    try {
      const resp = await importReferenceDataset({
        dataset_id: referenceDatasetId.trim(),
        ...(referenceImportMode === "registered" ? { reference_key: referenceKey } : {}),
        ...(referenceImportMode === "custom" ? { repo_id: referenceRepoId.trim() } : {}),
        ...(referenceImportMode === "local" ? { path: referenceLocalPath.trim() } : {}),
        ...(referenceImportMode !== "registered"
          ? {
              ...(referenceKey.trim() ? { reference_key: referenceKey.trim() } : {}),
              ...(referenceLicense.trim() ? { license: referenceLicense.trim() } : {}),
              note_field: referenceNoteField.trim(),
              ...(referenceQuestionField.trim()
                ? { question_field: referenceQuestionField.trim() }
                : {}),
              ...(referenceAnswerField.trim()
                ? { answer_field: referenceAnswerField.trim() }
                : {}),
              ...(referenceTaskField.trim() ? { task_field: referenceTaskField.trim() } : {}),
              ...(referencePatientIdField.trim()
                ? { patient_id_field: referencePatientIdField.trim() }
                : {}),
              ...(referenceImageField.trim() ? { image_field: referenceImageField.trim() } : {}),
              ...(referenceImageLabelField.trim()
                ? { image_label_field: referenceImageLabelField.trim() }
                : {}),
              ...(referenceLabValuesField.trim()
                ? { lab_values_field: referenceLabValuesField.trim() }
                : {}),
              ...(referenceVitalValuesField.trim()
                ? { vital_values_field: referenceVitalValuesField.trim() }
                : {}),
              ...(referenceMedicationsField.trim()
                ? { medications_field: referenceMedicationsField.trim() }
                : {}),
              ...(referenceTimeSeriesField.trim()
                ? { time_series_field: referenceTimeSeriesField.trim() }
                : {}),
            }
          : {}),
        ...(referenceSplit.trim() ? { split: referenceSplit.trim() } : {}),
        ...(parsedLimit !== undefined ? { limit: parsedLimit } : {}),
      });
      setReferenceImportResult(resp);
    } catch (err) {
      setReferenceError(err instanceof Error ? err.message : "Reference import failed");
    } finally {
      setIsImportingReference(false);
    }
  };

  const handleSyntheaImport = async () => {
    if (!syntheaPath.trim() || !syntheaDatasetId.trim() || isImportingSynthea) {
      return;
    }
    setSyntheaImportResult(null);
    setSyntheaError(null);
    setIsImportingSynthea(true);
    try {
      const resp = await importSyntheaFhir({
        path: syntheaPath.trim(),
        dataset_id: syntheaDatasetId.trim(),
      });
      setSyntheaImportResult(resp);
    } catch (err) {
      setSyntheaError(err instanceof Error ? err.message : "Synthea import failed");
    } finally {
      setIsImportingSynthea(false);
    }
  };

  const selectedReference = referenceCatalog.find((item) => item.key === referenceKey);
  const selectedRecipe = capabilities?.generation_recipes.find((item) => item.name === recipe);
  const includesClinicalText = modalities.includes("clinical_text");
  const includesImaging = modalities.includes("imaging");
  const includesTimeSeries = modalities.includes("time_series");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Generate Dataset</h1>
        <p className="mt-1 text-sm text-gray-600">
          Create synthetic healthcare records with structured EHR fields, notes, labs, vitals,
          time series, and generated or placeholder imaging assets.
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
        {capabilities && capabilities.generation_recipes.length > 0 && (
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Recipe</span>
              <select
                aria-label="Generation recipe"
                value={recipe}
                onChange={(event) => setRecipe(event.target.value)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              >
                <option value="">Manual configuration</option>
                {capabilities.generation_recipes.map((item) => (
                  <option key={item.name} value={item.name}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            {selectedRecipe && (
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-gray-600">
                <span className="rounded-md bg-white px-2 py-1">
                  {selectedRecipe.complexity}
                </span>
                <span className="rounded-md bg-white px-2 py-1">
                  {selectedRecipe.modalities.length} modalities
                </span>
                <span className="rounded-md bg-white px-2 py-1">
                  {selectedRecipe.export_formats.join(", ")}
                </span>
                <span>{selectedRecipe.description}</span>
                {selectedRecipe.recommended_reference_keys.length > 0 && (
                  <span className="basis-full text-gray-500">
                    Benchmark references:{" "}
                    {selectedRecipe.recommended_reference_keys.join(", ")} | thresholds{" "}
                    {selectedRecipe.benchmark_thresholds.min_overall_score}/
                    {selectedRecipe.benchmark_thresholds.min_metric_score}
                  </span>
                )}
              </div>
            )}
          </div>
        )}
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

        <div className="grid gap-4 md:grid-cols-[repeat(3,minmax(0,12rem))_minmax(0,1fr)_minmax(0,18rem)]">
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
            <span>Encounters</span>
            <input
              id="dataset-encounter-count"
              aria-label="Encounter count"
              type="number"
              value={encounterCount}
              onChange={(e) => setEncounterCount(e.target.value)}
              min={1}
              max={30}
              placeholder="1"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Topic mix</span>
            <input
              id="dataset-topic-mix"
              aria-label="Topic mix"
              type="text"
              value={topicMix}
              onChange={(e) => setTopicMix(e.target.value)}
              placeholder="sepsis, pneumonia, heart failure"
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

        <div className="grid gap-4 md:grid-cols-3">
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Race mix</span>
            <input
              id="dataset-races"
              aria-label="Race mix"
              type="text"
              value={races}
              onChange={(e) => setRaces(e.target.value)}
              placeholder="synthetic_white, synthetic_black"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Ethnicity mix</span>
            <input
              id="dataset-ethnicities"
              aria-label="Ethnicity mix"
              type="text"
              value={ethnicities}
              onChange={(e) => setEthnicities(e.target.value)}
              placeholder="synthetic_not_hispanic_or_latino"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Insurance mix</span>
            <input
              id="dataset-insurance"
              aria-label="Insurance mix"
              type="text"
              value={insurance}
              onChange={(e) => setInsurance(e.target.value)}
              placeholder="synthetic_medicare, synthetic_private"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Smoking mix</span>
            <input
              id="dataset-smoking-statuses"
              aria-label="Smoking mix"
              type="text"
              value={smokingStatuses}
              onChange={(e) => setSmokingStatuses(e.target.value)}
              placeholder="never, former, current"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Alcohol use mix</span>
            <input
              id="dataset-alcohol-use"
              aria-label="Alcohol use mix"
              type="text"
              value={alcoholUse}
              onChange={(e) => setAlcoholUse(e.target.value)}
              placeholder="none, occasional"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Housing mix</span>
            <input
              id="dataset-housing"
              aria-label="Housing mix"
              type="text"
              value={housing}
              onChange={(e) => setHousing(e.target.value)}
              placeholder="stable, unstable"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-sm font-medium text-gray-700">Exports</span>
          {exportFormatOptions.map((option) => {
            const selected = exportFormats.includes(option.value);
            return (
              <button
                key={option.value}
                type="button"
                onClick={() => toggleExportFormat(option.value)}
                aria-pressed={selected}
                className={`rounded-md border px-3 py-2 text-sm ${
                  selected
                    ? "border-gray-900 bg-gray-900 text-white"
                    : "border-gray-300 text-gray-600 hover:bg-gray-50"
                }`}
              >
                {option.label}
              </button>
            );
          })}
        </div>

        {includesClinicalText && (
          <div className="grid gap-4 md:grid-cols-[minmax(0,12rem)_minmax(0,12rem)_minmax(0,12rem)_minmax(0,16rem)_minmax(0,1fr)]">
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Text backend</span>
              <select
                aria-label="Clinical text backend"
                value={clinicalTextBackend}
                onChange={(event) =>
                  setClinicalTextBackend(
                    event.target.value as "deterministic" | "llm" | "external"
                  )
                }
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              >
                <option value="deterministic">Deterministic</option>
                <option value="llm">LLM</option>
                <option value="external">External</option>
              </select>
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Noise</span>
              <select
                aria-label="Clinical text noise profile"
                value={clinicalTextNoiseProfile}
                onChange={(event) =>
                  setClinicalTextNoiseProfile(event.target.value as ClinicalTextNoiseProfile)
                }
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              >
                {clinicalTextNoiseOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Profile</span>
              <select
                aria-label="Clinical text model profile"
                value={clinicalTextProfile}
                onChange={(event) => setClinicalTextProfile(event.target.value)}
                disabled={clinicalTextBackend !== "external"}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              >
                <option value="">Config default</option>
                {(capabilities?.clinical_text_model_profiles ?? []).map((profile) => (
                  <option key={profile.name} value={profile.name}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Command</span>
              <input
                aria-label="Clinical text external command"
                type="text"
                value={clinicalTextCommand}
                onChange={(event) => setClinicalTextCommand(event.target.value)}
                disabled={clinicalTextBackend !== "external"}
                placeholder="hf-note-sample,--model,local-notes"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Provider</span>
              <select
                aria-label="Clinical text LLM provider"
                value={llmProvider}
                onChange={(event) => setLlmProvider(event.target.value as LlmProviderOption)}
                disabled={clinicalTextBackend !== "llm"}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              >
                {llmProviderOptions.map((provider) => (
                  <option key={provider} value={provider}>
                    {provider}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Model</span>
              <input
                aria-label="Clinical text LLM model"
                type="text"
                value={llmModel}
                onChange={(event) => setLlmModel(event.target.value)}
                disabled={clinicalTextBackend !== "llm"}
                placeholder="Use config default"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700 md:col-span-2">
              <span>Ollama URL</span>
              <input
                aria-label="Ollama base URL"
                type="text"
                value={ollamaBaseUrl}
                onChange={(event) => setOllamaBaseUrl(event.target.value)}
                disabled={clinicalTextBackend !== "llm" || llmProvider !== "ollama"}
                placeholder="Use config default"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              />
            </label>
          </div>
        )}

        {includesImaging && (
          <div className="grid gap-4 md:grid-cols-[minmax(0,12rem)_minmax(0,18rem)_minmax(0,1fr)_minmax(0,1fr)]">
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Imaging backend</span>
              <select
                aria-label="Imaging backend"
                value={imagingBackend}
                onChange={(event) =>
                  setImagingBackend(
                    event.target.value as "placeholder" | "diffusers" | "external"
                  )
                }
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              >
                <option value="placeholder">Placeholder</option>
                <option value="diffusers">Diffusers</option>
                <option value="external">External</option>
              </select>
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Model profile</span>
              <select
                aria-label="Imaging model profile"
                value={imagingProfile}
                onChange={(event) => setImagingProfile(event.target.value)}
                disabled={imagingBackend !== "diffusers"}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              >
                <option value="">Config default</option>
                {(capabilities?.imaging_model_profiles ?? []).map((profile) => (
                  <option key={profile.name} value={profile.name}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Diffusers model id</span>
              <input
                aria-label="Diffusers model id"
                type="text"
                value={diffusersModelId}
                onChange={(event) => setDiffusersModelId(event.target.value)}
                disabled={imagingBackend !== "diffusers"}
                placeholder="Use profile or config default"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>External command</span>
              <input
                aria-label="Imaging external command"
                type="text"
                value={imagingCommand}
                onChange={(event) => setImagingCommand(event.target.value)}
                disabled={imagingBackend !== "external"}
                placeholder="hf-image-sample,--profile,cxr"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              />
            </label>
          </div>
        )}

        {includesTimeSeries && (
          <div className="grid gap-4 md:grid-cols-[minmax(0,12rem)_minmax(0,16rem)_minmax(0,1fr)]">
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Time-series backend</span>
              <select
                aria-label="Time-series backend"
                value={timeSeriesBackend}
                onChange={(event) =>
                  setTimeSeriesBackend(event.target.value as "deterministic" | "external")
                }
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              >
                <option value="deterministic">Deterministic</option>
                <option value="external">External</option>
              </select>
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Model profile</span>
              <select
                aria-label="Time-series model profile"
                value={timeSeriesProfile}
                onChange={(event) => setTimeSeriesProfile(event.target.value)}
                disabled={timeSeriesBackend !== "external"}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              >
                <option value="">Config default</option>
                {(capabilities?.time_series_model_profiles ?? []).map((profile) => (
                  <option key={profile.name} value={profile.name}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>External command</span>
              <input
                aria-label="Time-series external command"
                type="text"
                value={timeSeriesCommand}
                onChange={(event) => setTimeSeriesCommand(event.target.value)}
                disabled={timeSeriesBackend !== "external"}
                placeholder="timediff-sample,--checkpoint,local.pt"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal disabled:opacity-50"
              />
            </label>
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-[minmax(0,12rem)_minmax(0,12rem)_minmax(0,12rem)]">
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Fixture limit</span>
            <input
              aria-label="Release fixture limit"
              type="number"
              value={releaseFixtureLimit}
              onChange={(event) => setReleaseFixtureLimit(event.target.value)}
              min={1}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Overall gate</span>
            <input
              aria-label="Release minimum overall score"
              type="number"
              value={releaseMinOverallScore}
              onChange={(event) => setReleaseMinOverallScore(event.target.value)}
              min={0}
              max={1}
              step={0.01}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Metric gate</span>
            <input
              aria-label="Release minimum metric score"
              type="number"
              value={releaseMinMetricScore}
              onChange={(event) => setReleaseMinMetricScore(event.target.value)}
              min={0}
              max={1}
              step={0.01}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
        </div>

        <div className="grid gap-4 md:grid-cols-[minmax(0,12rem)_minmax(0,12rem)_minmax(0,12rem)]">
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Train split</span>
            <input
              aria-label="Release train split ratio"
              type="number"
              value={releaseTrainRatio}
              onChange={(event) => setReleaseTrainRatio(event.target.value)}
              min={0}
              step={0.01}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Validation split</span>
            <input
              aria-label="Release validation split ratio"
              type="number"
              value={releaseValidationRatio}
              onChange={(event) => setReleaseValidationRatio(event.target.value)}
              min={0}
              step={0.01}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Test split</span>
            <input
              aria-label="Release test split ratio"
              type="number"
              value={releaseTestRatio}
              onChange={(event) => setReleaseTestRatio(event.target.value)}
              min={0}
              step={0.01}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
        </div>

        <div>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={handleGenerate}
              disabled={
                !topic.trim() ||
                modalities.length === 0 ||
                exportFormats.length === 0 ||
                !Number.isInteger(count) ||
                count < 1 ||
                isGenerating
              }
              className="rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              Generate
            </button>
            <button
              type="button"
              onClick={handleGenerateReleasePackage}
              disabled={
                !topic.trim() ||
                modalities.length === 0 ||
                !Number.isInteger(count) ||
                count < 1 ||
                isGeneratingRelease
              }
              className="rounded-lg border border-blue-300 px-6 py-2 text-blue-700 hover:bg-blue-50 disabled:opacity-50"
            >
              Release Package
            </button>
          </div>
        </div>
      </div>

      {isGenerating && <div className="text-sm text-gray-600">Generating synthetic records...</div>}
      {isGeneratingRelease && (
        <div className="text-sm text-gray-600">Building release package...</div>
      )}

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

      {releaseResult && (
        <div className="rounded-lg border border-green-200 bg-green-50 p-4">
          <p className="font-medium text-green-800">Release package ready</p>
          <p className="text-sm text-green-700">{releaseResult.filename}</p>
          {releaseResult.datasetId && (
            <p className="text-xs text-green-700">{releaseResult.datasetId}</p>
          )}
          <button
            type="button"
            onClick={() => downloadBlob(releaseResult.blob, releaseResult.filename)}
            className="mt-3 rounded-md border border-green-300 bg-white px-3 py-2 text-sm font-medium text-green-800 hover:bg-green-100"
          >
            Download again
          </button>
        </div>
      )}

      {releaseError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="font-medium text-red-800">Release package failed</p>
          <p className="text-sm text-red-700">{releaseError}</p>
        </div>
      )}

      <div className="border-t border-gray-200 pt-6">
        <h2 className="text-xl font-semibold">Import Reference Dataset</h2>
        <p className="mt-1 text-sm text-gray-600">
          Pull registered, custom Hugging Face, or local JSON/JSONL reference
          datasets into the workbench for benchmarking and export.
        </p>

        <div className="mt-4 flex flex-wrap gap-2">
          {(["registered", "custom", "local"] as const).map((mode) => {
            const selected = referenceImportMode === mode;
            return (
              <button
                key={mode}
                type="button"
                onClick={() => setReferenceImportMode(mode)}
                aria-pressed={selected}
                className={`rounded-md border px-3 py-2 text-sm capitalize ${
                  selected
                    ? "border-gray-900 bg-gray-900 text-white"
                    : "border-gray-300 text-gray-600 hover:bg-gray-50"
                }`}
              >
                {mode}
              </button>
            );
          })}
        </div>

        <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,18rem)_minmax(0,16rem)_minmax(0,10rem)_minmax(0,8rem)]">
          {referenceImportMode === "registered" ? (
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Reference</span>
              <select
                aria-label="Reference dataset"
                value={referenceKey}
                onChange={(event) => setReferenceKey(event.target.value)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              >
                {referenceCatalog.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.key}
                  </option>
                ))}
              </select>
            </label>
          ) : referenceImportMode === "custom" ? (
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Repo id</span>
              <input
                aria-label="Custom Hugging Face repo id"
                type="text"
                value={referenceRepoId}
                onChange={(event) => setReferenceRepoId(event.target.value)}
                placeholder="org/custom-synthetic-notes"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
          ) : (
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Local path</span>
              <input
                aria-label="Local reference file path"
                type="text"
                value={referenceLocalPath}
                onChange={(event) => setReferenceLocalPath(event.target.value)}
                placeholder="/data/validation/local-notes.jsonl"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
          )}
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Dataset id</span>
            <input
              aria-label="Imported reference dataset id"
              type="text"
              value={referenceDatasetId}
              onChange={(event) => setReferenceDatasetId(event.target.value)}
              placeholder="ds-asclepius-ref"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Split</span>
            <input
              aria-label="Reference split"
              type="text"
              value={referenceSplit}
              onChange={(event) => setReferenceSplit(event.target.value)}
              placeholder={selectedReference?.split || "default"}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          {referenceImportMode !== "registered" && (
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>License</span>
              <input
                aria-label="Reference license"
                type="text"
                value={referenceLicense}
                onChange={(event) => setReferenceLicense(event.target.value)}
                placeholder="cc-by-4.0"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
          )}
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Limit</span>
            <input
              aria-label="Reference import limit"
              type="number"
              value={referenceLimit}
              onChange={(event) => setReferenceLimit(event.target.value)}
              min={1}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
        </div>

        {referenceImportMode !== "registered" && (
          <div className="mt-4 grid gap-4 md:grid-cols-[repeat(4,minmax(0,1fr))]">
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Reference key</span>
              <input
                aria-label="Reference key"
                type="text"
                value={referenceKey}
                onChange={(event) => setReferenceKey(event.target.value)}
                placeholder={
                  referenceImportMode === "local"
                    ? "local-validation-notes"
                    : "org/custom-synthetic-notes"
                }
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Note field</span>
              <input
                aria-label="Reference note field"
                type="text"
                value={referenceNoteField}
                onChange={(event) => setReferenceNoteField(event.target.value)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Prompt field</span>
              <input
                aria-label="Reference prompt field"
                type="text"
                value={referenceQuestionField}
                onChange={(event) => setReferenceQuestionField(event.target.value)}
                placeholder="prompt"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Answer field</span>
              <input
                aria-label="Reference answer field"
                type="text"
                value={referenceAnswerField}
                onChange={(event) => setReferenceAnswerField(event.target.value)}
                placeholder="completion"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Task field</span>
              <input
                aria-label="Reference task field"
                type="text"
                value={referenceTaskField}
                onChange={(event) => setReferenceTaskField(event.target.value)}
                placeholder="task"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Patient id field</span>
              <input
                aria-label="Reference patient id field"
                type="text"
                value={referencePatientIdField}
                onChange={(event) => setReferencePatientIdField(event.target.value)}
                placeholder="patient_id"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Image field</span>
              <input
                aria-label="Reference image field"
                type="text"
                value={referenceImageField}
                onChange={(event) => setReferenceImageField(event.target.value)}
                placeholder="image"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Image label field</span>
              <input
                aria-label="Reference image label field"
                type="text"
                value={referenceImageLabelField}
                onChange={(event) => setReferenceImageLabelField(event.target.value)}
                placeholder="label"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Labs field</span>
              <input
                aria-label="Reference labs field"
                type="text"
                value={referenceLabValuesField}
                onChange={(event) => setReferenceLabValuesField(event.target.value)}
                placeholder="labs"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Vitals field</span>
              <input
                aria-label="Reference vitals field"
                type="text"
                value={referenceVitalValuesField}
                onChange={(event) => setReferenceVitalValuesField(event.target.value)}
                placeholder="vitals"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Meds field</span>
              <input
                aria-label="Reference medications field"
                type="text"
                value={referenceMedicationsField}
                onChange={(event) => setReferenceMedicationsField(event.target.value)}
                placeholder="medications"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
            <label className="space-y-1 text-sm font-medium text-gray-700">
              <span>Series field</span>
              <input
                aria-label="Reference time series field"
                type="text"
                value={referenceTimeSeriesField}
                onChange={(event) => setReferenceTimeSeriesField(event.target.value)}
                placeholder="time_series"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
              />
            </label>
          </div>
        )}

        {referenceImportMode === "registered" && selectedReference && (
          <div className="mt-3 rounded-lg border border-gray-200 bg-gray-50 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium text-gray-900">
                {selectedReference.repo_id}
              </span>
              <span className="rounded-md bg-gray-200 px-2 py-1 text-xs text-gray-700">
                {selectedReference.license}
              </span>
              {selectedReference.gated && (
                <span className="rounded-md bg-yellow-100 px-2 py-1 text-xs text-yellow-800">
                  gated
                </span>
              )}
              <span className="rounded-md bg-blue-100 px-2 py-1 text-xs text-blue-700">
                {selectedReference.image_field
                  ? `${selectedReference.image_modality} ${selectedReference.image_body_region}`
                  : "text"}
              </span>
            </div>
            <p className="mt-2 text-xs text-gray-600">{selectedReference.description}</p>
            <p className="mt-1 text-xs text-gray-500">
              Use policy: {selectedReference.use_policy.replaceAll("_", " ")}
            </p>
          </div>
        )}

        {referenceImportMode === "local" && (
          <div className="mt-3 rounded-lg border border-blue-100 bg-blue-50 p-3 text-xs text-blue-900">
            Local imports accept JSON arrays, {"{rows: [...]}"}, JSONL, or NDJSON row
            files already accessible to the API process.
          </div>
        )}

        <div className="mt-4">
          <button
            type="button"
            onClick={handleReferenceImport}
            disabled={
              !referenceDatasetId.trim() ||
              isImportingReference ||
              (referenceImportMode === "registered" &&
                (!referenceKey || referenceCatalog.length === 0)) ||
              (referenceImportMode === "custom" &&
                (!referenceRepoId.trim() || !referenceNoteField.trim())) ||
              (referenceImportMode === "local" &&
                (!referenceLocalPath.trim() || !referenceNoteField.trim()))
            }
            className="rounded-lg bg-gray-900 px-6 py-2 text-white hover:bg-gray-800 disabled:opacity-50"
          >
            Import Reference
          </button>
        </div>

        {isImportingReference && (
          <div className="mt-4 text-sm text-gray-600">Importing reference records...</div>
        )}

        {referenceImportResult && (
          <div className="mt-4 rounded-lg border border-green-200 bg-green-50 p-4">
            <p className="font-medium text-green-800">Reference dataset imported</p>
            <p className="text-sm text-green-700">
              {referenceImportResult.imported} records from {referenceImportResult.reference_key}
            </p>
            <p className="text-xs text-green-700">{referenceImportResult.dataset_id}</p>
          </div>
        )}

        {referenceError && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-4">
            <p className="font-medium text-red-800">Reference import unavailable</p>
            <p className="text-sm text-red-700">{referenceError}</p>
          </div>
        )}
      </div>

      <div className="border-t border-gray-200 pt-6">
        <h2 className="text-xl font-semibold">Import Synthea FHIR</h2>
        <p className="mt-1 text-sm text-gray-600">
          Load Synthea FHIR JSON bundles from a file or directory into the dataset workbench.
        </p>

        <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,18rem)]">
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>FHIR path</span>
            <input
              aria-label="Synthea FHIR path"
              type="text"
              value={syntheaPath}
              onChange={(event) => setSyntheaPath(event.target.value)}
              placeholder="/path/to/synthea/output/fhir"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
          <label className="space-y-1 text-sm font-medium text-gray-700">
            <span>Dataset id</span>
            <input
              aria-label="Synthea dataset id"
              type="text"
              value={syntheaDatasetId}
              onChange={(event) => setSyntheaDatasetId(event.target.value)}
              placeholder="ds-synthea-cohort"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 font-normal"
            />
          </label>
        </div>

        <div className="mt-4">
          <button
            type="button"
            onClick={handleSyntheaImport}
            disabled={!syntheaPath.trim() || !syntheaDatasetId.trim() || isImportingSynthea}
            className="rounded-lg bg-gray-900 px-6 py-2 text-white hover:bg-gray-800 disabled:opacity-50"
          >
            Import Synthea
          </button>
        </div>

        {isImportingSynthea && (
          <div className="mt-4 text-sm text-gray-600">Importing Synthea records...</div>
        )}

        {syntheaImportResult && (
          <div className="mt-4 rounded-lg border border-green-200 bg-green-50 p-4">
            <p className="font-medium text-green-800">Synthea dataset imported</p>
            <p className="text-sm text-green-700">
              {syntheaImportResult.imported} records from {syntheaImportResult.source}
            </p>
            <p className="text-xs text-green-700">{syntheaImportResult.dataset_id}</p>
          </div>
        )}

        {syntheaError && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-4">
            <p className="font-medium text-red-800">Synthea import unavailable</p>
            <p className="text-sm text-red-700">{syntheaError}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
