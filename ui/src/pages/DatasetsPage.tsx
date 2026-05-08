import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  datasetExportUrl,
  fetchDataset,
  fetchDatasetBenchmark,
  fetchDatasetBenchmarkPlan,
  fetchDatasetBenchmarkSuite,
  fetchDatasetCard,
  fetchDatasetExports,
  fetchDatasetQuality,
  fetchDatasetReviewQueue,
  fetchDatasets,
  datasetImageUrl,
  saveRecordReview,
} from "../api/client";
import type {
  DatasetManifest,
  ExportFormat,
  ExportManifest,
  BenchmarkPlanReadiness,
  BenchmarkReport,
  BenchmarkSuiteReport,
  DatasetQualityReport,
  HumanReviewStatus,
  ReviewQueueItem,
  SyntheticRecordPreview,
} from "../api/client";

type RecipeBenchmarkPlan = {
  recipeName: string | null;
  recommendedReferenceKeys: string[];
  thresholds: {
    minOverallScore: number;
    minMetricScore: number;
  } | null;
};

export default function DatasetsPage() {
  const [topicFilter, setTopicFilter] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<ExportFormat>("sft_jsonl");
  const [includeReviewed, setIncludeReviewed] = useState(false);
  const [cardKind, setCardKind] = useState<"dataset" | "model">("dataset");
  const [referenceDatasetId, setReferenceDatasetId] = useState("");
  const [benchmarkMinOverallScore, setBenchmarkMinOverallScore] = useState(0.75);
  const [benchmarkMinMetricScore, setBenchmarkMinMetricScore] = useState(0.5);
  const queryClient = useQueryClient();

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => fetchDatasets(100),
  });
  const datasets = (data?.datasets ?? []).filter((dataset) =>
    dataset.topic.toLowerCase().includes(topicFilter.trim().toLowerCase())
  );
  const activeDatasetId = selectedDatasetId ?? datasets[0]?.dataset_id ?? null;

  const {
    data: detail,
    isLoading: isDetailLoading,
    isError: isDetailError,
    error: detailError,
    refetch: refetchDetail,
  } = useQuery({
    queryKey: ["dataset", activeDatasetId],
    queryFn: () => fetchDataset(activeDatasetId as string, 25),
    enabled: Boolean(activeDatasetId),
  });
  const exportFormats = detail?.manifest.export_formats ?? ["sft_jsonl"];
  const effectiveExportFormat = exportFormats.includes(exportFormat)
    ? exportFormat
    : (exportFormats[0] ?? "sft_jsonl");
  const benchmarkPlan = parseRecipeBenchmarkPlan(detail?.manifest.metadata);
  const hasAutoBenchmarkPlan =
    benchmarkPlan.recommendedReferenceKeys.length > 0 && benchmarkPlan.thresholds !== null;
  const {
    data: reviewQueue,
    isLoading: isReviewLoading,
    error: reviewError,
  } = useQuery({
    queryKey: ["dataset-reviews", activeDatasetId, includeReviewed],
    queryFn: () => fetchDatasetReviewQueue(activeDatasetId as string, includeReviewed, 100),
    enabled: Boolean(activeDatasetId),
  });
  const {
    data: quality,
    isLoading: isQualityLoading,
    error: qualityError,
  } = useQuery({
    queryKey: ["dataset-quality", activeDatasetId],
    queryFn: () => fetchDatasetQuality(activeDatasetId as string),
    enabled: Boolean(activeDatasetId),
  });
  const {
    data: exportAudit,
    isLoading: isExportAuditLoading,
    error: exportAuditError,
  } = useQuery({
    queryKey: ["dataset-exports", activeDatasetId],
    queryFn: () => fetchDatasetExports(activeDatasetId as string, 10),
    enabled: Boolean(activeDatasetId),
  });
  const {
    data: cardText,
    isLoading: isCardLoading,
    error: cardError,
  } = useQuery({
    queryKey: ["dataset-card", activeDatasetId, cardKind],
    queryFn: () => fetchDatasetCard(activeDatasetId as string, cardKind),
    enabled: Boolean(activeDatasetId),
  });
  const {
    data: benchmark,
    isLoading: isBenchmarkLoading,
    error: benchmarkError,
  } = useQuery({
    queryKey: [
      "dataset-benchmark",
      activeDatasetId,
      referenceDatasetId,
      benchmarkMinOverallScore,
      benchmarkMinMetricScore,
    ],
    queryFn: () =>
      fetchDatasetBenchmark(
        activeDatasetId as string,
        referenceDatasetId,
        benchmarkMinOverallScore,
        benchmarkMinMetricScore
      ),
    enabled: Boolean(activeDatasetId && referenceDatasetId),
  });
  const {
    data: benchmarkPlanReadiness,
    isLoading: isBenchmarkPlanLoading,
    error: benchmarkPlanError,
  } = useQuery({
    queryKey: ["dataset-benchmark-plan", activeDatasetId],
    queryFn: () => fetchDatasetBenchmarkPlan(activeDatasetId as string),
    enabled: Boolean(activeDatasetId),
  });
  const {
    data: benchmarkSuite,
    isLoading: isBenchmarkSuiteLoading,
    error: benchmarkSuiteError,
  } = useQuery({
    queryKey: ["dataset-benchmark-suite", activeDatasetId, benchmarkPlanReadiness?.ready],
    queryFn: () => fetchDatasetBenchmarkSuite(activeDatasetId as string),
    enabled: Boolean(activeDatasetId && benchmarkPlanReadiness?.ready),
  });
  const reviewMutation = useMutation({
    mutationFn: ({
      recordId,
      status,
    }: {
      recordId: string;
      status: HumanReviewStatus;
    }) =>
      saveRecordReview(recordId, {
        status,
        reviewer: "workbench",
        notes: [`Marked ${status} in dataset workbench.`],
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset-reviews", activeDatasetId] });
      queryClient.invalidateQueries({ queryKey: ["dataset-quality", activeDatasetId] });
      queryClient.invalidateQueries({ queryKey: ["dataset", activeDatasetId] });
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Dataset Workbench</h1>
        <p className="mt-1 text-sm text-gray-600">
          Review generated synthetic records, validation status, modalities, and fine-tuning exports.
        </p>
      </div>

      <input
        type="text"
        value={topicFilter}
        onChange={(e) => setTopicFilter(e.target.value)}
        placeholder="Filter by topic"
        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
      />

      {isLoading && <p className="text-sm text-gray-500">Loading...</p>}
      {Boolean(error) && (
        <div className="flex items-center justify-between rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="text-sm text-red-700">
            {error instanceof Error ? error.message : "Failed to load datasets."}
          </p>
          <button
            type="button"
            onClick={() => refetch()}
            className="rounded-md border border-red-300 px-3 py-1 text-sm text-red-700"
          >
            Retry
          </button>
        </div>
      )}

      {data && (
        <div className="grid gap-6 lg:grid-cols-[minmax(260px,320px)_1fr]">
          <div className="space-y-3">
            <p className="text-sm text-gray-500">{datasets.length} dataset(s)</p>
            {datasets.map((dataset) => (
              <DatasetRow
                key={dataset.dataset_id}
                dataset={dataset}
                selected={dataset.dataset_id === activeDatasetId}
                onSelect={() => setSelectedDatasetId(dataset.dataset_id)}
              />
            ))}
            {datasets.length === 0 && (
              <p className="rounded-lg border border-gray-200 p-4 text-sm text-gray-500">
                No synthetic datasets match this filter.
              </p>
            )}
          </div>

          <div className="space-y-4">
            {isDetailLoading && <p className="text-sm text-gray-500">Loading dataset preview...</p>}
            {isDetailError && (
              <div className="flex items-center justify-between rounded-lg border border-red-200 bg-red-50 p-4">
                <p className="text-sm text-red-700">
                  {detailError instanceof Error ? detailError.message : "Failed to load dataset preview."}
                </p>
                <button
                  type="button"
                  onClick={() => refetchDetail()}
                  className="rounded-md border border-red-300 px-3 py-1 text-sm text-red-700"
                >
                  Retry
                </button>
              </div>
            )}
            {detail && (
              <>
                <div className="rounded-lg border border-gray-200 bg-white p-4">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <p className="text-lg font-semibold text-gray-900">{detail.manifest.name}</p>
                      <p className="text-sm text-gray-500">{detail.manifest.dataset_id}</p>
                    </div>
                    <div className="flex gap-2">
                      <select
                        value={effectiveExportFormat}
                        onChange={(event) => setExportFormat(event.target.value as ExportFormat)}
                        className="rounded-md border border-gray-300 px-3 py-2 text-sm"
                      >
                        {exportFormats.map((format) => (
                          <option key={format} value={format}>
                            {format.replace("_", " ").toUpperCase()}
                          </option>
                        ))}
                      </select>
                      <a
                        href={datasetExportUrl(
                          detail.manifest.dataset_id,
                          effectiveExportFormat,
                          referenceDatasetId
                            ? {
                                referenceDatasetId,
                                minOverallScore: benchmarkMinOverallScore,
                                minMetricScore: benchmarkMinMetricScore,
                              }
                            : undefined
                        )}
                        className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
                      >
                        {referenceDatasetId ? "Export Gated" : "Export"}
                      </a>
                      {hasAutoBenchmarkPlan &&
                        !referenceDatasetId &&
                        benchmarkPlanReadiness?.ready && (
                          <a
                            href={datasetExportUrl(
                              detail.manifest.dataset_id,
                              effectiveExportFormat,
                              {
                                autoBenchmark: true,
                              }
                            )}
                            className="rounded-md border border-blue-300 px-4 py-2 text-sm font-medium text-blue-700 hover:bg-blue-50"
                          >
                            Export Auto-Gated
                          </a>
                        )}
                    </div>
                  </div>
                  <div className="mt-4 grid gap-3 sm:grid-cols-3">
                    <Metric label="Records" value={detail.manifest.generated_count} />
                    <Metric label="Approved" value={detail.manifest.approved_count} />
                    <Metric
                      label="Approval"
                      value={`${Math.round(
                        (detail.manifest.approved_count /
                          Math.max(detail.manifest.generated_count, 1)) *
                          100
                      )}%`}
                    />
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {detail.manifest.modalities.map((modality) => (
                      <span
                        key={modality}
                        className="rounded-md bg-gray-100 px-2 py-1 text-xs font-medium text-gray-700"
                      >
                        {modality.replace("_", " ")}
                      </span>
                    ))}
                  </div>
                </div>

                <QualityPanel
                  quality={quality ?? null}
                  isLoading={isQualityLoading}
                  error={qualityError}
                />

                <section className="grid gap-4 xl:grid-cols-[minmax(320px,0.9fr)_1.1fr]">
                  <ReviewQueuePanel
                    items={reviewQueue?.records ?? []}
                    isLoading={isReviewLoading}
                    error={reviewError}
                    includeReviewed={includeReviewed}
                    onIncludeReviewedChange={setIncludeReviewed}
                    onMark={(recordId, status) => reviewMutation.mutate({ recordId, status })}
                    pendingRecordId={
                      reviewMutation.isPending ? reviewMutation.variables?.recordId : null
                    }
                    mutationError={reviewMutation.error}
                  />
                  <CardPanel
                    cardKind={cardKind}
                    onCardKindChange={setCardKind}
                    cardText={cardText ?? ""}
                    isLoading={isCardLoading}
                    error={cardError}
                  />
                </section>

                <BenchmarkPanel
                  datasets={datasets}
                  activeDatasetId={detail.manifest.dataset_id}
                  referenceDatasetId={referenceDatasetId}
                  onReferenceDatasetChange={setReferenceDatasetId}
                  recipeBenchmarkPlan={benchmarkPlan}
                  benchmarkPlanReadiness={benchmarkPlanReadiness ?? null}
                  isBenchmarkPlanLoading={isBenchmarkPlanLoading}
                  benchmarkPlanError={benchmarkPlanError}
                  benchmarkSuite={benchmarkSuite ?? null}
                  isBenchmarkSuiteLoading={isBenchmarkSuiteLoading}
                  benchmarkSuiteError={benchmarkSuiteError}
                  benchmark={benchmark ?? null}
                  isLoading={isBenchmarkLoading}
                  error={benchmarkError}
                  minOverallScore={benchmarkMinOverallScore}
                  minMetricScore={benchmarkMinMetricScore}
                  onMinOverallScoreChange={setBenchmarkMinOverallScore}
                  onMinMetricScoreChange={setBenchmarkMinMetricScore}
                />

                <ExportAuditPanel
                  exports={exportAudit?.exports ?? []}
                  isLoading={isExportAuditLoading}
                  error={exportAuditError}
                />

                <div className="space-y-3">
                  {detail.records.map((record) => (
                    <RecordPreview key={record.record_id} record={record} />
                  ))}
                  {detail.records.length === 0 && (
                    <p className="rounded-lg border border-gray-200 p-4 text-sm text-gray-500">
                      This dataset has no previewable records.
                    </p>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function BenchmarkPanel({
  datasets,
  activeDatasetId,
  referenceDatasetId,
  onReferenceDatasetChange,
  recipeBenchmarkPlan,
  benchmarkPlanReadiness,
  isBenchmarkPlanLoading,
  benchmarkPlanError,
  benchmarkSuite,
  isBenchmarkSuiteLoading,
  benchmarkSuiteError,
  benchmark,
  isLoading,
  error,
  minOverallScore,
  minMetricScore,
  onMinOverallScoreChange,
  onMinMetricScoreChange,
}: {
  datasets: DatasetManifest[];
  activeDatasetId: string;
  referenceDatasetId: string;
  onReferenceDatasetChange: (datasetId: string) => void;
  recipeBenchmarkPlan: RecipeBenchmarkPlan;
  benchmarkPlanReadiness: BenchmarkPlanReadiness | null;
  isBenchmarkPlanLoading: boolean;
  benchmarkPlanError: unknown;
  benchmarkSuite: BenchmarkSuiteReport | null;
  isBenchmarkSuiteLoading: boolean;
  benchmarkSuiteError: unknown;
  benchmark: BenchmarkReport | null;
  isLoading: boolean;
  error: unknown;
  minOverallScore: number;
  minMetricScore: number;
  onMinOverallScoreChange: (score: number) => void;
  onMinMetricScoreChange: (score: number) => void;
}) {
  const referenceOptions = datasets.filter(
    (dataset) => dataset.dataset_id !== activeDatasetId
  );
  const recommendedReferenceOptions = referenceOptions.filter((dataset) =>
    datasetMatchesRecommendedReference(dataset, recipeBenchmarkPlan.recommendedReferenceKeys)
  );
  const topMetrics = benchmark?.metrics.slice(0, 6) ?? [];
  const hasRecipeBenchmarkPlan =
    recipeBenchmarkPlan.recommendedReferenceKeys.length > 0 ||
    recipeBenchmarkPlan.thresholds !== null;
  const applyRecipeThresholds = () => {
    if (!recipeBenchmarkPlan.thresholds) return;
    onMinOverallScoreChange(recipeBenchmarkPlan.thresholds.minOverallScore);
    onMinMetricScoreChange(recipeBenchmarkPlan.thresholds.minMetricScore);
  };
  const selectRecommendedReference = () => {
    const [firstRecommended] = recommendedReferenceOptions;
    if (firstRecommended) onReferenceDatasetChange(firstRecommended.dataset_id);
  };
  const selectResolvedReference = () => {
    if (benchmarkPlanReadiness?.resolved_reference_dataset_id) {
      onReferenceDatasetChange(benchmarkPlanReadiness.resolved_reference_dataset_id);
    }
  };
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-gray-900">Benchmark comparison</p>
          <p className="text-xs text-gray-500">Compare against a stored reference dataset</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={referenceDatasetId}
            onChange={(event) => onReferenceDatasetChange(event.target.value)}
            className="min-w-64 rounded-md border border-gray-300 px-3 py-2 text-sm"
          >
            <option value="">Select reference dataset</option>
            {referenceOptions.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {formatReferenceOption(dataset, recipeBenchmarkPlan.recommendedReferenceKeys)}
              </option>
            ))}
          </select>
          <label className="text-xs text-gray-600">
            Overall
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={minOverallScore}
              onChange={(event) =>
                onMinOverallScoreChange(clampScore(event.target.valueAsNumber))
              }
              className="ml-2 w-20 rounded-md border border-gray-300 px-2 py-2 text-sm"
            />
          </label>
          <label className="text-xs text-gray-600">
            Metric
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={minMetricScore}
              onChange={(event) =>
                onMinMetricScoreChange(clampScore(event.target.valueAsNumber))
              }
              className="ml-2 w-20 rounded-md border border-gray-300 px-2 py-2 text-sm"
            />
          </label>
        </div>
      </div>
      {hasRecipeBenchmarkPlan && (
        <div className="mt-3 rounded-md border border-blue-100 bg-blue-50 p-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase text-blue-800">
                Recipe benchmark plan
              </p>
              <p className="mt-1 text-sm text-blue-950">
                {recipeBenchmarkPlan.recipeName
                  ? recipeBenchmarkPlan.recipeName.replaceAll("_", " ")
                  : "Manifest recommendation"}
              </p>
              {recipeBenchmarkPlan.recommendedReferenceKeys.length > 0 && (
                <p className="mt-1 break-words text-xs text-blue-800">
                  References: {recipeBenchmarkPlan.recommendedReferenceKeys.join(", ")}
                </p>
              )}
              {recipeBenchmarkPlan.thresholds && (
                <p className="mt-1 text-xs text-blue-800">
                  Thresholds: overall{" "}
                  {formatMetricValue(recipeBenchmarkPlan.thresholds.minOverallScore)}, metric{" "}
                  {formatMetricValue(recipeBenchmarkPlan.thresholds.minMetricScore)}
                </p>
              )}
              {isBenchmarkPlanLoading && (
                <p className="mt-2 text-xs text-blue-800">Checking imported references...</p>
              )}
              {Boolean(benchmarkPlanError) && (
                <p className="mt-2 text-xs text-red-700">
                  {benchmarkPlanError instanceof Error
                    ? benchmarkPlanError.message
                    : "Failed to load benchmark plan readiness."}
                </p>
              )}
              {benchmarkPlanReadiness && (
                <div className="mt-2 space-y-1 text-xs text-blue-800">
                  <p>
                    Readiness:{" "}
                    <span className="font-medium">
                      {benchmarkPlanReadiness.ready ? "ready" : "missing reference import"}
                    </span>
                  </p>
                  {benchmarkPlanReadiness.resolved_reference_dataset_id && (
                    <p className="break-words">
                      Resolved reference:{" "}
                      {benchmarkPlanReadiness.resolved_reference_key ?? "unknown"} |{" "}
                      {benchmarkPlanReadiness.resolved_reference_dataset_id}
                    </p>
                  )}
                  {!benchmarkPlanReadiness.ready &&
                    benchmarkPlanReadiness.missing_reference_keys.length > 0 && (
                      <p className="break-words">
                        Missing imports: {benchmarkPlanReadiness.missing_reference_keys.join(", ")}
                      </p>
                    )}
                </div>
              )}
            </div>
            {recipeBenchmarkPlan.thresholds && (
              <button
                type="button"
                onClick={applyRecipeThresholds}
                className="shrink-0 rounded-md border border-blue-300 bg-white px-3 py-2 text-xs font-medium text-blue-800 hover:bg-blue-100"
              >
                Apply thresholds
              </button>
            )}
            {recommendedReferenceOptions.length > 0 && (
              <button
                type="button"
                onClick={selectRecommendedReference}
                className="shrink-0 rounded-md border border-blue-300 bg-white px-3 py-2 text-xs font-medium text-blue-800 hover:bg-blue-100"
              >
                Select reference
              </button>
            )}
            {benchmarkPlanReadiness?.resolved_reference_dataset_id && (
              <button
                type="button"
                onClick={selectResolvedReference}
                className="shrink-0 rounded-md border border-blue-300 bg-white px-3 py-2 text-xs font-medium text-blue-800 hover:bg-blue-100"
              >
                Use resolved
              </button>
            )}
          </div>
        </div>
      )}
      {referenceOptions.length === 0 && (
        <p className="mt-3 rounded-md bg-gray-50 p-3 text-sm text-gray-500">
          Import or generate another dataset to use as a benchmark reference.
        </p>
      )}
      {(isBenchmarkSuiteLoading || benchmarkSuiteError || benchmarkSuite) && (
        <div className="mt-4 rounded-md border border-gray-200 bg-gray-50 p-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase text-gray-500">
                Recipe benchmark suite
              </p>
              <p className="mt-1 text-sm text-gray-700">
                {isBenchmarkSuiteLoading
                  ? "Running recommended reference comparisons..."
                  : benchmarkSuite
                    ? `${benchmarkSuite.reference_count} imported reference comparison(s)`
                    : "Suite unavailable"}
              </p>
            </div>
            {benchmarkSuite && (
              <span
                className={`rounded-md px-2 py-1 text-xs font-medium ${
                  benchmarkSuite.passed
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {benchmarkSuite.passed ? "passed" : "failed"}
              </span>
            )}
          </div>
          {Boolean(benchmarkSuiteError) && (
            <p className="mt-2 text-sm text-red-700">
              {benchmarkSuiteError instanceof Error
                ? benchmarkSuiteError.message
                : "Failed to run benchmark suite."}
            </p>
          )}
          {benchmarkSuite && (
            <div className="mt-3 space-y-3">
              <div className="flex flex-wrap gap-3">
                <Metric
                  label="Mean score"
                  value={benchmarkSuite.mean_overall_score.toFixed(3)}
                />
                <Metric label="References" value={benchmarkSuite.reference_count} />
                <Metric
                  label="Suite gate"
                  value={benchmarkSuite.passed ? "passed" : "failed"}
                />
              </div>
              <p className="text-xs text-gray-500">
                Requires overall &gt;= {formatMetricValue(benchmarkSuite.thresholds.min_overall_score)} and each metric &gt;={" "}
                {formatMetricValue(benchmarkSuite.thresholds.min_metric_score)}.
              </p>
              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                {benchmarkSuite.results.map((result) => (
                  <div
                    key={`${result.reference_key}-${result.reference_dataset_id}`}
                    className="rounded-md border border-gray-200 bg-white p-3"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-gray-900">
                          {result.reference_key}
                        </p>
                        <p className="mt-1 break-words text-xs text-gray-500">
                          {result.reference_dataset_id}
                        </p>
                      </div>
                      <span
                        className={`shrink-0 rounded-md px-2 py-1 text-xs font-medium ${
                          result.passed
                            ? "bg-green-50 text-green-700"
                            : "bg-red-50 text-red-700"
                        }`}
                      >
                        {result.passed ? "passed" : "failed"}
                      </span>
                    </div>
                    <p className="mt-2 text-lg font-semibold text-gray-900">
                      {result.overall_score.toFixed(3)}
                    </p>
                    {result.failing_metrics.length > 0 && (
                      <p className="mt-1 line-clamp-2 text-xs text-red-700">
                        {result.failing_metrics.slice(0, 4).join(", ")}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {isLoading && <p className="mt-3 text-sm text-gray-500">Running benchmark...</p>}
      {Boolean(error) && (
        <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error instanceof Error ? error.message : "Failed to run benchmark."}
        </p>
      )}
      {benchmark && (
        <div className="mt-4 space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <Metric label="Overall score" value={benchmark.overall_score.toFixed(3)} />
            <Metric label="Gate" value={benchmark.passed ? "passed" : "failed"} />
            <Metric label="Metric count" value={benchmark.metrics.length} />
            <Metric label="Failing metrics" value={benchmark.failing_metrics.length} />
            <Metric label="Warnings" value={benchmark.warnings.length} />
          </div>
          <p className="text-xs text-gray-500">
            Requires overall &gt;= {formatMetricValue(benchmark.thresholds.min_overall_score)} and each metric &gt;={" "}
            {formatMetricValue(benchmark.thresholds.min_metric_score)}.
          </p>
          {benchmark.failing_metrics.length > 0 && (
            <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">
              <p className="font-medium">Failing benchmark metrics</p>
              <p className="mt-1">{benchmark.failing_metrics.join(", ")}</p>
            </div>
          )}
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {topMetrics.map((metric) => (
              <div key={metric.name} className="rounded-md border border-gray-200 p-3">
                <p className="text-xs font-medium uppercase text-gray-500">
                  {metric.name.replaceAll("_", " ")}
                </p>
                <p className="mt-1 text-lg font-semibold text-gray-900">
                  {metric.score.toFixed(3)}
                </p>
                <p className="mt-1 text-xs text-gray-500">
                  gen {formatMetricValue(metric.generated_value)} | ref{" "}
                  {formatMetricValue(metric.reference_value)}
                </p>
              </div>
            ))}
          </div>
          {benchmark.warnings.length > 0 && (
            <div className="rounded-md bg-yellow-50 p-3 text-sm text-yellow-800">
              {benchmark.warnings.slice(0, 3).map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ExportAuditPanel({
  exports,
  isLoading,
  error,
}: {
  exports: ExportManifest[];
  isLoading: boolean;
  error: unknown;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-gray-900">Export audit trail</p>
          <p className="text-xs text-gray-500">Recent fine-tuning exports and benchmark gates</p>
        </div>
        <span className="rounded-md bg-gray-100 px-2 py-1 text-xs text-gray-700">
          {exports.length} export(s)
        </span>
      </div>
      {isLoading && <p className="mt-3 text-sm text-gray-500">Loading export audit trail...</p>}
      {Boolean(error) && (
        <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error instanceof Error ? error.message : "Failed to load export audit trail."}
        </p>
      )}
      {!isLoading && !error && exports.length === 0 && (
        <p className="mt-3 rounded-md bg-gray-50 p-3 text-sm text-gray-500">
          No exports have been recorded for this dataset.
        </p>
      )}
      {exports.length > 0 && (
        <div className="mt-4 overflow-auto">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead>
              <tr className="text-left text-xs font-medium uppercase text-gray-500">
                <th className="py-2 pr-4">Format</th>
                <th className="py-2 pr-4">Records</th>
                <th className="py-2 pr-4">Gate</th>
                <th className="py-2 pr-4">Reference</th>
                <th className="py-2 pr-4">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {exports.map((item) => (
                <tr key={`${item.created_at}-${item.export_format}-${item.file_path}`}>
                  <td className="py-2 pr-4 font-medium text-gray-900">
                    {item.export_format.replace("_", " ")}
                  </td>
                  <td className="py-2 pr-4 text-gray-600">{item.record_count}</td>
                  <td className="py-2 pr-4">
                    <span className={exportGateClass(item.metadata.benchmark_passed)}>
                      {formatExportGate(item.metadata.benchmark_passed)}
                    </span>
                  </td>
                  <td className="py-2 pr-4 text-gray-600">
                    {formatMetadataValue(item.metadata.benchmark_reference_dataset_id)}
                  </td>
                  <td className="py-2 pr-4 text-gray-600">
                    {formatTimestamp(item.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function clampScore(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function parseRecipeBenchmarkPlan(
  metadata: Record<string, unknown> | undefined
): RecipeBenchmarkPlan {
  if (!metadata) {
    return {
      recipeName: null,
      recommendedReferenceKeys: [],
      thresholds: null,
    };
  }
  return {
    recipeName: stringFromMetadata(metadata.primary_recipe),
    recommendedReferenceKeys: stringListFromMetadata(metadata.recommended_reference_keys),
    thresholds: benchmarkThresholdsFromMetadata(metadata.benchmark_thresholds),
  };
}

function stringFromMetadata(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function stringListFromMetadata(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean);
}

function benchmarkThresholdsFromMetadata(
  value: unknown
): RecipeBenchmarkPlan["thresholds"] {
  if (!value || typeof value !== "object") return null;
  const thresholds = value as Record<string, unknown>;
  const minOverallScore = scoreFromMetadata(thresholds.min_overall_score);
  const minMetricScore = scoreFromMetadata(thresholds.min_metric_score);
  if (minOverallScore === null || minMetricScore === null) return null;
  return { minOverallScore, minMetricScore };
}

function scoreFromMetadata(value: unknown): number | null {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  return clampScore(value);
}

function datasetMatchesRecommendedReference(
  dataset: DatasetManifest,
  recommendedReferenceKeys: string[]
) {
  const primaryReferenceKey = stringFromMetadata(dataset.metadata.primary_reference_key);
  return Boolean(
    primaryReferenceKey && recommendedReferenceKeys.includes(primaryReferenceKey)
  );
}

function formatReferenceOption(
  dataset: DatasetManifest,
  recommendedReferenceKeys: string[]
) {
  const primaryReferenceKey = stringFromMetadata(dataset.metadata.primary_reference_key);
  const labelParts = [dataset.topic, dataset.dataset_id];
  if (primaryReferenceKey) {
    labelParts.push(`ref ${primaryReferenceKey}`);
  }
  const label = labelParts.join(" | ");
  if (
    primaryReferenceKey &&
    recommendedReferenceKeys.includes(primaryReferenceKey)
  ) {
    return `${label} | recommended`;
  }
  return label;
}

function QualityPanel({
  quality,
  isLoading,
  error,
}: {
  quality: DatasetQualityReport | null;
  isLoading: boolean;
  error: unknown;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-gray-900">Fine-tuning readiness</p>
          <p className="text-xs text-gray-500">Validation and export quality gate</p>
        </div>
        {quality && (
          <span
            className={`rounded-md px-2 py-1 text-xs font-medium ${
              quality.export_ready
                ? "bg-green-50 text-green-700"
                : "bg-yellow-50 text-yellow-700"
            }`}
          >
            {quality.export_ready ? "Ready" : "Blocked"}
          </span>
        )}
      </div>
      {isLoading && <p className="mt-3 text-sm text-gray-500">Loading quality report...</p>}
      {Boolean(error) && (
        <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error instanceof Error ? error.message : "Failed to load quality report."}
        </p>
      )}
      {quality && (
        <div className="mt-4 space-y-3">
          <div className="grid gap-3 sm:grid-cols-4">
            <Metric label="Records" value={quality.record_count} />
            <Metric label="Approved" value={quality.approved_count} />
            <Metric label="Approval" value={`${Math.round(quality.approval_rate * 100)}%`} />
            <Metric label="Blockers" value={quality.blocking_issue_count} />
            <Metric
              label="Alignment"
              value={
                quality.mean_modality_alignment_score === null ||
                quality.mean_modality_alignment_score === undefined
                  ? "none"
                  : quality.mean_modality_alignment_score.toFixed(2)
              }
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-4">
            <Metric label="Documents" value={quality.artifact_counts.documents ?? 0} />
            <Metric label="Diagnoses" value={quality.artifact_counts.diagnoses ?? 0} />
            <Metric label="Labs" value={quality.artifact_counts.labs ?? 0} />
            <Metric label="Vitals" value={quality.artifact_counts.vitals ?? 0} />
            <Metric label="Medications" value={quality.artifact_counts.medications ?? 0} />
            <Metric label="Procedures" value={quality.artifact_counts.procedures ?? 0} />
            <Metric label="Series" value={quality.artifact_counts.time_series_channels ?? 0} />
            <Metric label="Images" value={quality.artifact_counts.imaging_assets ?? 0} />
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Metric
              label="Fact keys"
              value={Object.keys(quality.extracted_fact_key_counts).length}
            />
            <Metric
              label="Series backends"
              value={Object.keys(quality.time_series_backend_counts).length}
            />
            <Metric
              label="Image backends"
              value={Object.keys(quality.imaging_backend_counts).length}
            />
            <Metric
              label="Image policies"
              value={Object.keys(quality.imaging_model_policy_counts).length}
            />
            <Metric
              label="Code systems"
              value={Object.keys(quality.diagnosis_code_system_counts).length}
            />
            <Metric
              label="Diagnosis codes"
              value={Object.keys(quality.diagnosis_code_counts).length}
            />
            <Metric label="PHI entities" value={Object.keys(quality.phi_entity_counts).length} />
            <Metric
              label="Lab signals"
              value={Object.keys(quality.lab_numeric_summaries).length}
            />
            <Metric
              label="Vital signals"
              value={Object.keys(quality.vital_numeric_summaries).length}
            />
            <Metric
              label="Series signals"
              value={Object.keys(quality.time_series_numeric_summaries).length}
            />
          </div>
          {quality.recommended_reference_keys.length > 0 && (
            <div className="rounded-md border border-blue-100 bg-blue-50 p-3 text-sm text-blue-900">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-medium">
                  Benchmark references: {quality.recommended_reference_keys.join(", ")}
                </span>
                <span
                  className={`rounded-md px-2 py-1 text-xs font-medium ${
                    quality.benchmark_ready
                      ? "bg-green-100 text-green-700"
                      : "bg-yellow-100 text-yellow-800"
                  }`}
                >
                  {quality.benchmark_ready ? "Reference ready" : "Reference missing"}
                </span>
              </div>
              {quality.resolved_reference_dataset_id && (
                <p className="mt-1 break-words text-xs text-blue-800">
                  Resolved: {quality.resolved_reference_dataset_id}
                </p>
              )}
              {!quality.benchmark_ready && quality.missing_reference_keys.length > 0 && (
                <p className="mt-1 break-words text-xs text-blue-800">
                  Missing imports: {quality.missing_reference_keys.join(", ")}
                </p>
              )}
            </div>
          )}
          {quality.recommendations.length > 0 && (
            <div className="rounded-md bg-yellow-50 p-3 text-sm text-yellow-800">
              {quality.recommendations.slice(0, 3).map((recommendation) => (
                <p key={recommendation}>{recommendation}</p>
              ))}
            </div>
          )}
          {Object.keys(quality.note_type_counts).length > 0 && (
            <div className="flex flex-wrap gap-2">
              {Object.entries(quality.note_type_counts)
                .slice(0, 6)
                .map(([noteType, count]) => (
                  <span
                    key={noteType}
                    className="rounded-md bg-blue-50 px-2 py-1 text-xs text-blue-700"
                  >
                    {noteType.replace("_", " ")}: {count}
                  </span>
                ))}
            </div>
          )}
          {Object.keys(quality.extracted_fact_key_counts).length > 0 && (
            <div className="flex flex-wrap gap-2">
              {Object.entries(quality.extracted_fact_key_counts)
                .slice(0, 6)
                .map(([factKey, count]) => (
                  <span
                    key={factKey}
                    className="rounded-md bg-green-50 px-2 py-1 text-xs text-green-700"
                  >
                    {factKey.replace("_", " ")}: {count}
                  </span>
                ))}
            </div>
          )}
          {Object.keys(quality.issue_counts_by_field).length > 0 && (
            <div className="flex flex-wrap gap-2">
              {Object.entries(quality.issue_counts_by_field)
                .slice(0, 6)
                .map(([field, count]) => (
                  <span
                    key={field}
                    className="rounded-md bg-gray-100 px-2 py-1 text-xs text-gray-700"
                  >
                    {field}: {count}
                  </span>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ReviewQueuePanel({
  items,
  isLoading,
  error,
  includeReviewed,
  onIncludeReviewedChange,
  onMark,
  pendingRecordId,
  mutationError,
}: {
  items: ReviewQueueItem[];
  isLoading: boolean;
  error: unknown;
  includeReviewed: boolean;
  onIncludeReviewedChange: (value: boolean) => void;
  onMark: (recordId: string, status: HumanReviewStatus) => void;
  pendingRecordId: string | null;
  mutationError: unknown;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-gray-900">Human review queue</p>
          <p className="text-xs text-gray-500">{items.length} record(s)</p>
        </div>
        <label className="flex items-center gap-2 text-xs text-gray-600">
          <input
            type="checkbox"
            checked={includeReviewed}
            onChange={(event) => onIncludeReviewedChange(event.target.checked)}
            className="h-4 w-4 rounded border-gray-300"
          />
          Include reviewed
        </label>
      </div>
      {isLoading && <p className="mt-3 text-sm text-gray-500">Loading review queue...</p>}
      {Boolean(error) && (
        <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error instanceof Error ? error.message : "Failed to load review queue."}
        </p>
      )}
      {Boolean(mutationError) && (
        <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {mutationError instanceof Error ? mutationError.message : "Failed to save review."}
        </p>
      )}
      <div className="mt-3 max-h-96 space-y-2 overflow-auto pr-1">
        {items.map((item) => (
          <ReviewQueueRow
            key={item.record_id}
            item={item}
            isPending={pendingRecordId === item.record_id}
            onMark={onMark}
          />
        ))}
        {!isLoading && items.length === 0 && (
          <p className="rounded-md bg-gray-50 p-3 text-sm text-gray-500">
            No records currently require human review.
          </p>
        )}
      </div>
    </div>
  );
}

function ReviewQueueRow({
  item,
  isPending,
  onMark,
}: {
  item: ReviewQueueItem;
  isPending: boolean;
  onMark: (recordId: string, status: HumanReviewStatus) => void;
}) {
  const status = item.human_review?.status ?? "pending";
  return (
    <div className="rounded-md border border-gray-200 p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-sm font-medium text-gray-900">{item.record_id}</p>
          <p className="mt-1 text-xs text-gray-500">
            {item.complexity} | validation{" "}
            {item.validation_approved === null || item.validation_approved === undefined
              ? "unknown"
              : item.validation_approved
                ? "approved"
                : "blocked"}{" "}
            | issues {item.issue_count}
          </p>
        </div>
        <span className={reviewStatusClass(status)}>{status.replace("_", " ")}</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          disabled={isPending}
          onClick={() => onMark(item.record_id, "approved")}
          className="rounded-md bg-green-600 px-3 py-1 text-xs font-medium text-white disabled:opacity-50"
        >
          Approve
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() => onMark(item.record_id, "needs_revision")}
          className="rounded-md bg-yellow-500 px-3 py-1 text-xs font-medium text-white disabled:opacity-50"
        >
          Revise
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() => onMark(item.record_id, "rejected")}
          className="rounded-md bg-red-600 px-3 py-1 text-xs font-medium text-white disabled:opacity-50"
        >
          Reject
        </button>
      </div>
    </div>
  );
}

function CardPanel({
  cardKind,
  onCardKindChange,
  cardText,
  isLoading,
  error,
}: {
  cardKind: "dataset" | "model";
  onCardKindChange: (kind: "dataset" | "model") => void;
  cardText: string;
  isLoading: boolean;
  error: unknown;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-gray-900">Generated cards</p>
          <p className="text-xs text-gray-500">Dataset and model documentation</p>
        </div>
        <div className="flex rounded-md border border-gray-300 p-1">
          {(["dataset", "model"] as const).map((kind) => (
            <button
              key={kind}
              type="button"
              onClick={() => onCardKindChange(kind)}
              className={`rounded px-3 py-1 text-xs font-medium ${
                cardKind === kind ? "bg-gray-900 text-white" : "text-gray-600"
              }`}
            >
              {kind === "dataset" ? "Dataset" : "Model"}
            </button>
          ))}
        </div>
      </div>
      {isLoading && <p className="mt-3 text-sm text-gray-500">Loading card...</p>}
      {Boolean(error) && (
        <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error instanceof Error ? error.message : "Failed to load card."}
        </p>
      )}
      {!isLoading && !error && (
        <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap rounded-md bg-gray-950 p-3 text-xs leading-relaxed text-gray-100">
          {cardText || "No card content available."}
        </pre>
      )}
    </div>
  );
}

function DatasetRow({
  dataset,
  selected,
  onSelect,
}: {
  dataset: DatasetManifest;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full rounded-lg border p-4 text-left transition-colors ${
        selected
          ? "border-blue-500 bg-blue-50"
          : "border-gray-200 bg-white hover:border-gray-300"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-medium text-gray-900">{dataset.topic}</p>
          <p className="mt-1 text-xs text-gray-500">{dataset.dataset_id}</p>
        </div>
        <span className="rounded-md bg-gray-100 px-2 py-1 text-xs text-gray-700">
          {dataset.generated_count}
        </span>
      </div>
      <div className="mt-3 flex flex-wrap gap-1">
        {dataset.modalities.slice(0, 4).map((modality) => (
          <span key={modality} className="text-xs text-gray-500">
            {modality.replace("_", " ")}
          </span>
        ))}
      </div>
    </button>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-lg border border-gray-200 p-3">
      <p className="text-xs font-medium uppercase text-gray-500">{label}</p>
      <p className="mt-1 text-xl font-semibold text-gray-900">{value}</p>
    </div>
  );
}

function reviewStatusClass(status: HumanReviewStatus) {
  const base = "rounded-md px-2 py-1 text-xs font-medium";
  if (status === "approved") return `${base} bg-green-50 text-green-700`;
  if (status === "rejected") return `${base} bg-red-50 text-red-700`;
  if (status === "needs_revision") return `${base} bg-yellow-50 text-yellow-700`;
  return `${base} bg-gray-100 text-gray-700`;
}

function formatMetricValue(value: number | string | null) {
  if (typeof value === "number") return value.toFixed(2);
  if (value === null) return "none";
  return value;
}

function formatMetadataValue(value: unknown) {
  if (typeof value === "string" && value.trim()) return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "none";
}

function formatExportGate(value: unknown) {
  if (value === true) return "passed";
  if (value === false) return "failed";
  return "not run";
}

function exportGateClass(value: unknown) {
  const base = "rounded-md px-2 py-1 text-xs font-medium";
  if (value === true) return `${base} bg-green-50 text-green-700`;
  if (value === false) return `${base} bg-red-50 text-red-700`;
  return `${base} bg-gray-100 text-gray-700`;
}

function formatTimestamp(value: string) {
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return value;
  return new Date(timestamp).toLocaleString();
}

function RecordPreview({ record }: { record: SyntheticRecordPreview }) {
  const note = record.documents[0];
  const scores = record.validation;
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-medium text-gray-900">{record.record_id}</p>
          <p className="text-sm text-gray-500">
            {record.patient.age} {record.patient.sex} | {record.complexity}
          </p>
        </div>
        {scores && (
          <span
            className={`rounded-md px-2 py-1 text-xs font-medium ${
              scores.approved ? "bg-green-50 text-green-700" : "bg-yellow-50 text-yellow-700"
            }`}
          >
            {scores.approved ? "Approved" : "Needs review"}
          </span>
        )}
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <p className="text-sm text-gray-600">
          Labs: {record.labs.map((lab) => `${lab.name} ${lab.value}${lab.unit}`).join(", ") || "none"}
        </p>
        <p className="text-sm text-gray-600">
          Vitals: {record.vitals.map((vital) => `${vital.name} ${vital.value}${vital.unit}`).join(", ") || "none"}
        </p>
        <p className="text-sm text-gray-600">
          Meds: {formatMedicationSummary(record.medication_history)}
        </p>
        <p className="text-sm text-gray-600">
          Assets: {record.documents.length} notes, {record.imaging.length} images
        </p>
      </div>
      {note && (
        <div className="mt-3 rounded-md bg-gray-50 p-3">
          <p className="text-xs font-medium uppercase text-gray-500">{note.note_type}</p>
          <p className="mt-1 line-clamp-3 text-sm text-gray-700">{note.clean_text}</p>
        </div>
      )}
      {record.imaging.length > 0 && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {record.imaging.slice(0, 3).map((image) => (
            <div key={image.image_id} className="overflow-hidden rounded-md border border-gray-200">
              {image.file_path ? (
                <img
                  src={datasetImageUrl(record.dataset_id, image.image_id)}
                  alt={`${image.modality} ${image.body_region}`}
                  className="aspect-video w-full bg-gray-100 object-contain"
                  loading="lazy"
                />
              ) : (
                <div className="flex aspect-video items-center justify-center bg-gray-100 text-xs text-gray-500">
                  No image file
                </div>
              )}
              <div className="p-2 text-xs text-gray-600">
                <p className="font-medium text-gray-900">
                  {image.modality} {image.body_region}
                </p>
                <p className="mt-1 truncate">{image.image_id}</p>
              </div>
            </div>
          ))}
        </div>
      )}
      {scores && (
        <div className="mt-3 flex flex-wrap gap-3 text-xs text-gray-500">
          <span>Schema {scores.schema_score.toFixed(2)}</span>
          <span>Consistency {scores.clinical_consistency_score.toFixed(2)}</span>
          <span>Privacy {scores.privacy_score.toFixed(2)}</span>
          <span>Utility {scores.utility_score.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}

function formatMedicationSummary(
  medications: SyntheticRecordPreview["medication_history"]
) {
  if (!medications.length) return "none";
  return medications
    .slice(0, 3)
    .map((medication) =>
      [medication.name, medication.dose, medication.route]
        .filter(Boolean)
        .join(" ")
    )
    .join(", ");
}
