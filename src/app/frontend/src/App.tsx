import {
  Suspense,
  lazy,
  useState,
  useEffect,
  useCallback,
  useId,
  useMemo,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import Editor from "@monaco-editor/react";
import {
  fetchLoadTestSample,
  fetchLoadTestRun,
  fetchActiveLoadTestRuns,
  fetchModelSettings,
  fetchModels,
  fetchPresets,
  cancelLoadTestRun,
  calibrateLoadTestOutputEstimate,
  fetchLoadTestOutputEstimate,
  runInference,
  startLoadTest,
} from "./api";
import type {
  ApiError,
  LoadTestResponse,
  LoadTestOutputEstimateRequest,
  LoadTestOutputEstimateResponse,
  LoadTestRun,
  LoadTestSampleResponse,
  ModelInfo,
  Preset,
  ModelSlot,
} from "./types";
import {
  buildLoadTestRequest,
  DEFAULT_REFERENCE_ITPM,
  DEFAULT_REFERENCE_OTPM,
  DEFAULT_REFERENCE_QPH,
  LOAD_TEST_PROMPT_MAX_LENGTH,
  requestLatencyPercentile,
  type LoadTestFieldErrors,
  type LoadTestFormValues,
} from "./loadTestUtils";
import SchemaBuilder from "./SchemaBuilder";
import PresetManager from "./PresetManager";
import ModelSettingsManager from "./ModelSettingsManager";
import HelpDrawer from "./HelpDrawer";
import ModelResultColumn from "./ModelResultColumn";

const LoadTestCharts = lazy(() => import("./LoadTestCharts"));

let slotCounter = 0;
const LEFT_PANE_MODE_KEY = "clinical-extractor:left-pane-mode";
const ACTIVE_VIEW_KEY = "clinical-extractor:active-view";
const SYSTEM_PROMPT_KEY = "clinical-extractor:system-prompt";
const SYSTEM_PROMPT_VISIBILITY_KEY = "clinical-extractor:show-system-prompt";
const MAX_OUTPUT_TOKENS_KEY = "clinical-extractor:max-output-tokens";
const ACTIVE_LOAD_TEST_RUN_KEY = "clinical-extractor:active-load-test-run";
const SYSTEM_PROMPT_MAX_LENGTH = 4000;
const MEDICAL_TEXT_MAX_LENGTH = 10000;
const DEFAULT_SCHEMA = `{
  "type": "object",
  "properties": {}
}`;
// Upper bound the app allows for a per-request output cap. Must not exceed the
// serving endpoint's max_model_len. Empty override falls back to the
// per-endpoint default configured in models.json.
const MAX_OUTPUT_TOKENS_CEILING = 8192;
const DEFAULT_SYSTEM_PROMPT =
  "You are a clinical information extraction assistant. Extract only information explicitly supported by the provided medical text. Return output that matches the requested schema exactly. Do not invent facts; when the source text does not provide a value, leave it null or empty if the schema allows.";
const MODEL_UNAVAILABLE_MESSAGE =
  "This model is not currently verified as ready. Refresh its status or choose another model.";
const CUSTOM_MODEL_WARMUP_MESSAGE =
  "This custom endpoint may still be starting up. Try again in a moment.";
const CUSTOM_MODEL_TIMEOUT_MESSAGE =
  "This custom endpoint timed out before it finished responding. Try a smaller prompt or schema, lower max output tokens, or retry after checking endpoint health.";
const NO_MODELS_AVAILABLE_MESSAGE =
  "No models are currently available. Try again.";
const CUSTOM_MODEL_WARMUP_ERROR_CODES = new Set(["ENDPOINT_ERROR", "ENDPOINT_NOT_FOUND"]);

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  if (error && typeof error === "object" && "message" in error) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }
  return fallback;
}

function makeSlotId() {
  return `slot-${++slotCounter}`;
}

type LeftPaneMode = "collapsed" | "normal" | "wide";
type AppView = "structured-output" | "load-testing";

const APP_VIEW_OPTIONS: Array<{
  id: AppView;
  title: string;
}> = [
  {
    id: "structured-output",
    title: "Playground",
  },
  {
    id: "load-testing",
    title: "Load Testing",
  },
];
const DEFAULT_LOAD_TEST_CONFIG = {
  total_requests: 150,
  max_workers: 10,
  max_retries: 10,
  max_tokens: 4096,
  cache_bust: true,
};
const LOAD_TEST_MAX_REQUESTS = 150;
const LOAD_TEST_MAX_WORKERS = 10;
const LOAD_TEST_MAX_TOKENS = 4096;
const LOAD_TEST_MAX_TARGET_TPM = 10_000_000;
const DEFAULT_LOAD_TEST_FORM = {
  schedulingMode: "burst" as const,
  totalRequests: String(DEFAULT_LOAD_TEST_CONFIG.total_requests),
  durationSeconds: "60",
  loadMultiplier: "1",
  expectedOutputTokens: "",
  maxWorkers: String(DEFAULT_LOAD_TEST_CONFIG.max_workers),
  maxTokens: String(DEFAULT_LOAD_TEST_CONFIG.max_tokens),
  targetItpm: String(DEFAULT_REFERENCE_ITPM),
  targetOtpm: String(DEFAULT_REFERENCE_OTPM),
  prompt: "",
  schemaJson: "",
  systemPrompt: "",
};

type LoadTestFormState = LoadTestFormValues;

function isLoadTestRunLive(run: LoadTestRun | null): boolean {
  return run?.status === "queued" || run?.status === "running";
}

function HelpIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <circle cx="8" cy="8" r="5.75" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M6.45 6.35A1.65 1.65 0 0 1 8.1 4.8c1.05 0 1.75.63 1.75 1.55 0 1.82-1.95 1.7-1.95 3.05"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.4"
      />
      <circle cx="7.9" cy="11.55" r="0.75" fill="currentColor" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        d="M2.25 4.25h3M8.75 4.25h5M2.25 8h6.5M12.25 8h1.5M2.25 11.75h1.5M7.25 11.75h6.5"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.4"
      />
      <circle cx="7" cy="4.25" r="1.25" fill="none" stroke="currentColor" strokeWidth="1.35" />
      <circle cx="10.5" cy="8" r="1.25" fill="none" stroke="currentColor" strokeWidth="1.35" />
      <circle cx="5.5" cy="11.75" r="1.25" fill="none" stroke="currentColor" strokeWidth="1.35" />
    </svg>
  );
}

function ThemeIcon({ theme }: { theme: "light" | "dark" }) {
  return theme === "light" ? (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        d="M11.9 10.4A5.2 5.2 0 0 1 5.6 4.1 5.2 5.2 0 1 0 11.9 10.4Z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.45"
      />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <circle cx="8" cy="8" r="2.6" fill="none" stroke="currentColor" strokeWidth="1.45" />
      <path
        d="M8 1.5v1.3M8 13.2v1.3M1.5 8h1.3M13.2 8h1.3M3.4 3.4l.9.9M11.7 11.7l.9.9M12.6 3.4l-.9.9M4.3 11.7l-.9.9"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.35"
      />
    </svg>
  );
}

function FocusIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        d="M6 2.5H2.5V6M10 2.5h3.5V6M6 13.5H2.5V10M10 13.5h3.5V10"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.4"
      />
    </svg>
  );
}

function formatCompactInteger(value: number | null | undefined): string {
  if (value == null) {
    return "—";
  }
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null) {
    return "—";
  }
  if (value >= 100) {
    return `${Math.round(value)}s`;
  }
  return `${value.toFixed(1)}s`;
}

function formatRetryHistory(
  events: LoadTestResponse["requests"][number]["retry_events"]
): string {
  if (!events?.length) return "—";
  return events
    .map((event) => {
      const reason = event.status_code
        ? `HTTP ${event.status_code}`
        : event.error_code ?? "invalid response";
      const retryAfter =
        event.retry_after_s == null ? "" : `, Retry-After ${formatSeconds(event.retry_after_s)}`;
      return `#${event.attempt} ${reason} at ${formatSeconds(event.elapsed_s)} → ${formatSeconds(event.backoff_s)} backoff${retryAfter}`;
    })
    .join("; ");
}

function formatPercentage(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

function formatTargetSource(value: string | null | undefined): string {
  if (!value || value === "none") return "No reference";
  return value === "documented" ? "Endpoint reference" : "User-entered reference";
}

function formatOtpm(value: number | null | undefined): string {
  if (value == null) {
    return "—";
  }
  return `${formatCompactInteger(value)} OTPM`;
}

function formatTokenRate(
  value: number | null | undefined,
  kind: "ITPM" | "OTPM"
): string {
  if (value == null) {
    return "—";
  }
  return `${formatCompactInteger(value)} ${kind}`;
}

function formatLimitValue(value: number | null | undefined): string {
  if (value == null) {
    return "—";
  }
  return new Intl.NumberFormat("en-US").format(value);
}

function parseIntegerInput(value: string): number | null {
  const trimmed = value.trim();
  if (!/^-?\d+$/.test(trimmed)) {
    return null;
  }

  const parsed = Number(trimmed);
  return Number.isSafeInteger(parsed) ? parsed : null;
}

function PreviewChevron({ expanded }: { expanded: boolean }) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        d={expanded ? "M3.75 9.75L8 6l4.25 3.75" : "M3.75 6.25L8 10l4.25-3.75"}
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.75"
      />
    </svg>
  );
}

type LoadTestingViewProps = {
  theme: "light" | "dark";
  models: ModelInfo[];
  selectedModelInfo: ModelInfo | null;
  selectedModel: string;
  onModelChange: (model: string) => void;
  form: LoadTestFormState;
  onFormChange: (field: keyof LoadTestFormState, value: string) => void;
  onResetTarget: () => void;
  showTargetReset: boolean;
  onRun: () => void;
  onCancel: () => void;
  onRestoreSample: () => void;
  onUsePlayground: () => void;
  outputEstimate: LoadTestOutputEstimateResponse | null;
  estimateLoading: boolean;
  calibrationLoading: boolean;
  estimateError: string;
  manualOutputEstimate: boolean;
  onCalibrateOutput: () => void;
  onManualOutputEstimateChange: (manual: boolean) => void;
  loading: boolean;
  cancelling: boolean;
  activeRun: LoadTestRun | null;
  result: LoadTestResponse | null;
  error: ApiError | null;
  fieldErrors: LoadTestFieldErrors;
  completedAt: Date | null;
  sample: LoadTestSampleResponse | null;
  sampleError: string;
  showSamplePreview: boolean;
  onToggleSamplePreview: () => void;
};

function LoadTestingView({
  theme,
  models,
  selectedModelInfo,
  selectedModel,
  onModelChange,
  form,
  onFormChange,
  onResetTarget,
  showTargetReset,
  onRun,
  onCancel,
  onRestoreSample,
  onUsePlayground,
  outputEstimate,
  estimateLoading,
  calibrationLoading,
  estimateError,
  manualOutputEstimate,
  onCalibrateOutput,
  onManualOutputEstimateChange,
  loading,
  cancelling,
  activeRun,
  result,
  error,
  fieldErrors,
  completedAt,
  sample,
  sampleError,
  showSamplePreview,
  onToggleSamplePreview,
}: LoadTestingViewProps) {
  const noModels = models.length === 0;
  const busy = loading || cancelling;
  const summary = result?.summary;
  const resultModeLabel = result?.config.scheduling_mode === "target_rate" ? "target-rate run" : "burst run";
  const p95Latency =
    summary?.p95_latency_s ??
    (result ? requestLatencyPercentile(result.requests, 95) : null);
  const inputTargetLabel =
    summary?.target_itpm != null
      ? `${result?.config.target_itpm_source ?? result?.config.target_source ?? "custom"} ITPM reference`
      : "Input reference";
  const outputTargetLabel =
    summary?.target_otpm != null
      ? `${result?.config.target_otpm_source ?? result?.config.target_source ?? "custom"} OTPM reference`
      : "Output reference";
  const resultDisplayName =
    result && selectedModelInfo?.name === result.model
      ? selectedModelInfo.rate_limits?.docs_model_name ?? result.model
      : result?.model ?? "Endpoint";
  const activeRunDisplayName =
    activeRun && selectedModelInfo?.name === activeRun.model
      ? selectedModelInfo.rate_limits?.docs_model_name ?? activeRun.model
      : activeRun?.model ?? "Endpoint";
  const activeRunIsLive = isLoadTestRunLive(activeRun);
  const progressTotal = activeRun?.progress.total_requests ?? 0;
  const progressCompleted = activeRun?.progress.completed_requests ?? 0;
  const progressPercent =
    progressTotal > 0 ? Math.min(100, (progressCompleted / progressTotal) * 100) : 0;
  const expectedOutputTokens = parseIntegerInput(form.expectedOutputTokens);
  const targetOtpm = parseIntegerInput(form.targetOtpm);
  const durationSeconds = parseIntegerInput(form.durationSeconds);
  const loadMultiplier = Number(form.loadMultiplier);
  const otpmRequestsPerMinute =
    expectedOutputTokens && targetOtpm && Number.isFinite(loadMultiplier)
      ? (targetOtpm / expectedOutputTokens) * loadMultiplier
      : null;
  const otpmIntervalSeconds =
    otpmRequestsPerMinute && otpmRequestsPerMinute > 0
      ? 60 / otpmRequestsPerMinute
      : null;
  const otpmPlannedRequests =
    otpmIntervalSeconds && durationSeconds
      ? Math.floor(durationSeconds / otpmIntervalSeconds) + 1
      : null;
  const measuredAt = outputEstimate?.measured_at
    ? new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      }).format(new Date(outputEstimate.measured_at))
    : null;
  const needsOutputEstimate =
    form.schedulingMode === "target_rate" &&
    Boolean(form.targetOtpm.trim()) &&
    expectedOutputTokens == null;
  return (
    <main id="main-workspace" className="load-testing-view">
      <form
        className="load-test-toolbar-card"
        onSubmit={(event) => {
          event.preventDefault();
          onRun();
        }}
        noValidate
      >
        <div className="load-test-toolbar-copy">
          <span className="load-testing-eyebrow">Load Testing</span>
          <h2>Endpoint token-throughput test</h2>
          <p>
            {form.schedulingMode === "burst"
              ? "Push the endpoint immediately to expose throttling and retry behavior."
              : "Schedule arrivals by wall clock at user-verified input and output token rates."}
          </p>
        </div>

        <div className="load-test-toolbar-actions">
          <div className="load-test-config-panel">
            <div className="load-test-config-grid">
              <label className="load-test-field wide">
                <span className="load-test-field-label">Endpoint</span>
                <select
                  id="load-test-model"
                  name="load-test-model"
                  value={selectedModel}
                  onChange={(event) => onModelChange(event.target.value)}
                  disabled={busy || noModels}
                  aria-label="Select endpoint for load test"
                  aria-describedby={fieldErrors.model ? "load-test-model-error" : undefined}
                  aria-invalid={Boolean(fieldErrors.model)}
                >
                  {models.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name}
                    </option>
                  ))}
                </select>
                {fieldErrors.model && (
                  <span id="load-test-model-error" className="load-test-field-error">
                    {fieldErrors.model}
                  </span>
                )}
              </label>

              <label className="load-test-field">
                <span className="load-test-field-label">Mode</span>
                <select
                  id="load-test-mode"
                  name="load-test-mode"
                  value={form.schedulingMode}
                  onChange={(event) => onFormChange("schedulingMode", event.target.value)}
                  disabled={busy || noModels}
                >
                  <option value="burst">Burst</option>
                  <option value="target_rate">Target rate</option>
                </select>
                <span className="load-test-field-help">
                  {form.schedulingMode === "burst" ? "Immediate saturation" : "Clock-driven arrivals"}
                </span>
              </label>

              {form.schedulingMode === "burst" ? <label className="load-test-field">
                <span className="load-test-field-label">Requests</span>
                <input
                  id="load-test-requests"
                  name="load-test-requests"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={LOAD_TEST_MAX_REQUESTS}
                  value={form.totalRequests}
                  onChange={(event) =>
                    onFormChange("totalRequests", event.target.value)
                  }
                  disabled={busy || noModels}
                  aria-describedby={
                    fieldErrors.totalRequests ? "load-test-requests-error" : "load-test-requests-help"
                  }
                  aria-invalid={Boolean(fieldErrors.totalRequests)}
                />
                <span id="load-test-requests-help" className="load-test-field-help">1–150 logical requests</span>
                {fieldErrors.totalRequests && (
                  <span id="load-test-requests-error" className="load-test-field-error">
                    {fieldErrors.totalRequests}
                  </span>
                )}
              </label> : <label className="load-test-field">
                <span className="load-test-field-label">Duration</span>
                <input
                  id="load-test-duration"
                  name="load-test-duration"
                  type="number"
                  inputMode="numeric"
                  min={10}
                  max={600}
                  value={form.durationSeconds}
                  onChange={(event) => onFormChange("durationSeconds", event.target.value)}
                  disabled={busy || noModels}
                  aria-invalid={Boolean(fieldErrors.durationSeconds)}
                />
                <span className="load-test-field-help">10–600 seconds</span>
                {fieldErrors.durationSeconds && <span className="load-test-field-error">{fieldErrors.durationSeconds}</span>}
              </label>}

              <label className="load-test-field">
                <span className="load-test-field-label">
                  {form.schedulingMode === "burst" ? "Workers" : "Max in flight"}
                </span>
                <input
                  id="load-test-workers"
                  name="load-test-workers"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={LOAD_TEST_MAX_WORKERS}
                  value={form.maxWorkers}
                  onChange={(event) => onFormChange("maxWorkers", event.target.value)}
                  disabled={busy || noModels}
                  aria-describedby={
                    fieldErrors.maxWorkers ? "load-test-workers-error" : "load-test-workers-help"
                  }
                  aria-invalid={Boolean(fieldErrors.maxWorkers)}
                />
                <span id="load-test-workers-help" className="load-test-field-help">
                  Safety ceiling: 1–10 calls
                </span>
                {fieldErrors.maxWorkers && (
                  <span id="load-test-workers-error" className="load-test-field-error">
                    {fieldErrors.maxWorkers}
                  </span>
                )}
              </label>

              <label className="load-test-field">
                <span className="load-test-field-label">{form.schedulingMode === "burst" ? "OTPM reference" : "Target OTPM"}</span>
                <input
                  id="load-test-target-otpm"
                  name="load-test-target-otpm"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={LOAD_TEST_MAX_TARGET_TPM}
                  placeholder={String(DEFAULT_REFERENCE_OTPM)}
                  value={form.targetOtpm}
                  onChange={(event) => onFormChange("targetOtpm", event.target.value)}
                  disabled={busy || noModels}
                  aria-describedby={
                    fieldErrors.targetOtpm ? "load-test-target-otpm-error" : "load-test-target-otpm-help"
                  }
                  aria-invalid={Boolean(fieldErrors.targetOtpm)}
                />
                <span id="load-test-target-otpm-help" className="load-test-field-help">
                  {form.schedulingMode === "burst" ? "Optional comparison only" : "Leave blank to pace by ITPM"}
                </span>
                {fieldErrors.targetOtpm && (
                  <span id="load-test-target-otpm-error" className="load-test-field-error">
                    {fieldErrors.targetOtpm}
                  </span>
                )}
              </label>

              {form.schedulingMode === "target_rate" && <>
                <label className="load-test-field">
                  <span className="load-test-field-label">Target ITPM</span>
                  <input
                    id="load-test-target-itpm"
                    name="load-test-target-itpm"
                    type="number"
                    inputMode="numeric"
                    min={1}
                    max={LOAD_TEST_MAX_TARGET_TPM}
                    placeholder={String(DEFAULT_REFERENCE_ITPM)}
                    value={form.targetItpm}
                    onChange={(event) => onFormChange("targetItpm", event.target.value)}
                    disabled={busy || noModels}
                    aria-invalid={Boolean(fieldErrors.targetItpm)}
                  />
                  <span className="load-test-field-help">Leave blank to pace by OTPM</span>
                  {fieldErrors.targetItpm && <span className="load-test-field-error">{fieldErrors.targetItpm}</span>}
                </label>
                {form.targetOtpm.trim() && (
                  <div className="load-test-field load-test-output-estimate" aria-live="polite">
                    <span className="load-test-field-label">Output tokens</span>
                    <div className="load-test-output-estimate-value">
                      <strong>
                        {manualOutputEstimate && expectedOutputTokens
                          ? expectedOutputTokens.toLocaleString()
                          : outputEstimate?.tokens
                            ? outputEstimate.tokens.toLocaleString()
                            : "—"}
                      </strong>
                    </div>
                    <div className="load-test-output-estimate-meta">
                      <span className="load-test-field-help">
                        {manualOutputEstimate
                          ? "Manual override"
                          : calibrationLoading
                            ? "Calibrating…"
                            : estimateLoading
                              ? "Checking history…"
                              : outputEstimate?.tokens
                                ? `${outputEstimate.source === "load_test" ? "Measured from load test" : "Calibrated"}${measuredAt ? ` · ${measuredAt}` : ""}`
                                : "Not measured"}
                      </span>
                      {!manualOutputEstimate && (
                        <button
                          type="button"
                          className="load-test-calibrate-button"
                          onClick={onCalibrateOutput}
                          disabled={busy || noModels || calibrationLoading}
                        >
                          {calibrationLoading
                            ? "Calibrating…"
                            : outputEstimate?.tokens
                              ? "Recalibrate"
                              : "Calibrate"}
                        </button>
                      )}
                    </div>
                    {estimateError && <span className="load-test-field-error">{estimateError}</span>}
                    {fieldErrors.expectedOutputTokens && (
                      <span className="load-test-field-error">{fieldErrors.expectedOutputTokens}</span>
                    )}
                  </div>
                )}
                <label className="load-test-field">
                  <span className="load-test-field-label">Load</span>
                  <select
                    id="load-test-multiplier"
                    name="load-test-multiplier"
                    value={form.loadMultiplier}
                    onChange={(event) => onFormChange("loadMultiplier", event.target.value)}
                    disabled={busy || noModels}
                    aria-invalid={Boolean(fieldErrors.loadMultiplier)}
                  >
                    <option value="1">100% of target</option>
                    <option value="1.25">125% of target</option>
                    <option value="1.5">150% of target</option>
                  </select>
                  <span className="load-test-field-help">Use above 100% to provoke throttling</span>
                </label>
              </>}
            </div>

            <div className="load-test-estimate-line" role="status" aria-live="polite">
              <strong>
                {form.schedulingMode === "burst"
                  ? `${form.totalRequests || "—"} ${parseIntegerInput(form.totalRequests) === 1 ? "request" : "requests"}`
                  : `${form.durationSeconds || "—"} second target-rate run`}
              </strong>
              <span aria-hidden="true">·</span>
              <span>
                {form.maxWorkers || "—"}{" "}
                {parseIntegerInput(form.maxWorkers) === 1 ? "worker" : "workers"}
              </span>
              <span aria-hidden="true">·</span>
              <span>{form.schedulingMode === "burst" ? "burst scheduling" : `${form.loadMultiplier || "—"}× target load`}</span>
              <span aria-hidden="true">·</span>
              <span>{form.targetOtpm.trim() ? `${formatCompactInteger(parseIntegerInput(form.targetOtpm))} OTPM ${form.schedulingMode === "burst" ? "reference" : "target"}` : "OTPM unrestricted"}</span>
              {form.schedulingMode === "target_rate" && otpmRequestsPerMinute && otpmIntervalSeconds && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>
                    OTPM plan: ~{otpmRequestsPerMinute.toFixed(1)} starts/min · every {otpmIntervalSeconds.toFixed(1)}s
                    {otpmPlannedRequests ? ` · up to ${otpmPlannedRequests} starts` : ""}
                    {form.targetItpm.trim() ? " · ITPM may be stricter" : ""}
                  </span>
                </>
              )}
            </div>

            {Object.values(fieldErrors).find(Boolean) && (
              <p className="load-test-inline-error" role="alert">
                {Object.values(fieldErrors).find(Boolean)}
              </p>
            )}

            <div className="load-test-run-actions">
              <button
                type="submit"
                className="btn-submit load-test-run-button"
                disabled={busy || noModels || !selectedModel || needsOutputEstimate}
                aria-busy={loading}
              >
                {loading && <span className="spinner" />}
                {activeRunIsLive
                  ? "Run in progress"
                  : loading
                    ? "Starting…"
                    : needsOutputEstimate
                      ? "Calibrate output first"
                      : "Run load test"}
              </button>
            </div>
          </div>
        </div>
      </form>

      <section className="load-test-sample-card">
        <button
          type="button"
          className="load-test-sample-toggle"
          onClick={onToggleSamplePreview}
          aria-expanded={showSamplePreview}
          aria-controls="load-test-workload-body"
        >
          <div className="load-test-sample-toggle-copy">
            <span className="load-testing-eyebrow">Advanced</span>
            <strong>References, output cap, and test payload</strong>
            <p>
              {sample?.name ?? "Clinical extraction payload"} · customize only when needed.
            </p>
          </div>
          <span className="load-test-sample-toggle-icon">
            <PreviewChevron expanded={showSamplePreview} />
          </span>
        </button>

        {showSamplePreview && (
          <div id="load-test-workload-body" className="load-test-sample-body">
            {sampleError && (
              <div className="error">
                <div className="error-title">{sampleError}</div>
              </div>
            )}

            <div className="load-test-advanced-grid">
              {form.schedulingMode === "burst" && <label className="load-test-field">
                <span className="load-test-field-label">ITPM reference</span>
                <input
                  id="load-test-target-itpm"
                  name="load-test-target-itpm"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={LOAD_TEST_MAX_TARGET_TPM}
                  placeholder={String(DEFAULT_REFERENCE_ITPM)}
                  value={form.targetItpm}
                  onChange={(event) => onFormChange("targetItpm", event.target.value)}
                  disabled={busy || noModels}
                  aria-describedby={
                    fieldErrors.targetItpm ? "load-test-target-itpm-error" : "load-test-target-itpm-help"
                  }
                  aria-invalid={Boolean(fieldErrors.targetItpm)}
                />
                <span id="load-test-target-itpm-help" className="load-test-field-help">
                  Optional comparison; does not pace requests
                </span>
                {fieldErrors.targetItpm && (
                  <span id="load-test-target-itpm-error" className="load-test-field-error">
                    {fieldErrors.targetItpm}
                  </span>
                )}
              </label>}
              {form.schedulingMode === "target_rate" && form.targetOtpm.trim() && (
                <div className="load-test-output-override">
                  <div className="load-test-output-override-heading">
                    <span className="load-test-field-label">Output sizing</span>
                    <button
                      type="button"
                      className="load-test-reset-button"
                      onClick={() => onManualOutputEstimateChange(!manualOutputEstimate)}
                      disabled={busy}
                      aria-pressed={manualOutputEstimate}
                    >
                      {manualOutputEstimate ? "Use automatic" : "Enter manually"}
                    </button>
                  </div>
                  {manualOutputEstimate ? (
                    <label className="load-test-field">
                      <span className="load-test-field-label">Tokens per response</span>
                      <input
                        id="load-test-expected-output"
                        name="load-test-expected-output"
                        type="number"
                        inputMode="numeric"
                        min={1}
                        max={parseIntegerInput(form.maxTokens) ?? LOAD_TEST_MAX_TOKENS}
                        value={form.expectedOutputTokens}
                        onChange={(event) =>
                          onFormChange("expectedOutputTokens", event.target.value)
                        }
                        disabled={busy || noModels}
                        aria-invalid={Boolean(fieldErrors.expectedOutputTokens)}
                      />
                    </label>
                  ) : (
                    <p>Uses a saved measurement for this exact endpoint, prompt, schema, and output cap.</p>
                  )}
                </div>
              )}
              <label className="load-test-field">
                <span className="load-test-field-label">Max output tokens</span>
                <input
                  id="load-test-max-tokens"
                  name="load-test-max-tokens"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={
                    selectedModelInfo
                      ? Math.min(
                          selectedModelInfo.max_tokens ?? LOAD_TEST_MAX_TOKENS,
                          LOAD_TEST_MAX_TOKENS
                        )
                      : LOAD_TEST_MAX_TOKENS
                  }
                  value={form.maxTokens}
                  onChange={(event) => onFormChange("maxTokens", event.target.value)}
                  disabled={busy || noModels}
                  aria-invalid={Boolean(fieldErrors.maxTokens)}
                />
                <span className="load-test-field-help">Conservative output budget per request</span>
              </label>
              <div className="load-test-limits-copy">
                <span className="load-test-field-label">Starting references</span>
                <strong>Verify before running</strong>
                <p>
                  {`ITPM ${formatLimitValue(DEFAULT_REFERENCE_ITPM)} · OTPM ${formatLimitValue(DEFAULT_REFERENCE_OTPM)} · QPH ${formatLimitValue(DEFAULT_REFERENCE_QPH)}`}
                </p>
                <p>
                  These values are not inferred from the selected endpoint.{" "}
                  <a
                    href="https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits#rate-limits-by-model"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Verify in the Databricks reference
                  </a>
                  .
                </p>
                {showTargetReset && (
                  <button
                    type="button"
                    className="load-test-reset-button"
                    onClick={onResetTarget}
                    disabled={busy}
                  >
                    Restore starting references
                  </button>
                )}
              </div>
            </div>

            <div className="load-test-sample-meta">
              <button
                type="button"
                className="btn-secondary"
                onClick={onRestoreSample}
                disabled={busy || !sample}
              >
                Restore sample
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={onUsePlayground}
                disabled={busy}
              >
                Use Playground input
              </button>
              <span className="load-testing-pill">
                Cache-busting per request · up to 10 retries · 2–60s full jitter
              </span>
            </div>
            <div className="load-test-workload-grid">
              <label className="load-test-workload-field">
                <span className="load-test-field-label">Prompt</span>
                <textarea
                  id="load-test-prompt"
                  name="load-test-prompt"
                  className="load-test-prompt-input"
                  value={form.prompt}
                  onChange={(event) => onFormChange("prompt", event.target.value)}
                  disabled={busy}
                  maxLength={LOAD_TEST_PROMPT_MAX_LENGTH}
                  aria-invalid={Boolean(fieldErrors.prompt)}
                  aria-describedby="load-test-prompt-meta"
                />
                <span
                  id="load-test-prompt-meta"
                  className={fieldErrors.prompt ? "load-test-field-error" : "load-test-field-help"}
                >
                  {fieldErrors.prompt ?? `${form.prompt.length.toLocaleString()} / ${LOAD_TEST_PROMPT_MAX_LENGTH.toLocaleString()} characters`}
                </span>
              </label>
              <label className="load-test-workload-field">
                <span className="load-test-field-label">JSON response schema</span>
                <textarea
                  id="load-test-schema"
                  name="load-test-schema"
                  className="is-code"
                  value={form.schemaJson}
                  onChange={(event) => onFormChange("schemaJson", event.target.value)}
                  disabled={busy}
                  maxLength={20000}
                  spellCheck={false}
                  aria-invalid={Boolean(fieldErrors.schemaJson)}
                  aria-describedby="load-test-schema-meta"
                />
                <span
                  id="load-test-schema-meta"
                  className={fieldErrors.schemaJson ? "load-test-field-error" : "load-test-field-help"}
                >
                  {fieldErrors.schemaJson ?? `${form.schemaJson.length.toLocaleString()} / 20,000 characters`}
                </span>
              </label>
              <label className="load-test-workload-field full-width">
                <span className="load-test-field-label">System prompt</span>
                <textarea
                  id="load-test-system-prompt"
                  name="load-test-system-prompt"
                  value={form.systemPrompt}
                  onChange={(event) => onFormChange("systemPrompt", event.target.value)}
                  disabled={busy}
                  maxLength={4000}
                  aria-invalid={Boolean(fieldErrors.systemPrompt)}
                  aria-describedby="load-test-system-prompt-meta"
                />
                <span
                  id="load-test-system-prompt-meta"
                  className={fieldErrors.systemPrompt ? "load-test-field-error" : "load-test-field-help"}
                >
                  {fieldErrors.systemPrompt ?? `${form.systemPrompt.length.toLocaleString()} / 4,000 characters`}
                </span>
              </label>
            </div>
          </div>
        )}
      </section>

      {noModels && !activeRunIsLive && (
        <section className="load-test-message-card" role="status">
          <p>No ready configured endpoints are available in this workspace.</p>
        </section>
      )}

      {error && (
        <section className="load-test-message-card error" role="alert" aria-live="assertive">
          <div className="error-title">{error.message}</div>
          {error.detail && <div className="error-detail">{error.detail}</div>}
        </section>
      )}

      {summary && (
        <>
          <section className="load-test-outcome" aria-live="polite">
            <div>
              <span className="load-testing-eyebrow">Run outcome</span>
              <h2>
                {resultDisplayName} — achieved {formatOtpm(summary.achieved_otpm)}
                {summary.target_otpm == null
                  ? ""
                  : ` vs ${formatCompactInteger(summary.target_otpm)} OTPM reference`}
              </h2>
              <p>
                {completedAt
                  ? `Completed ${completedAt.toLocaleString()} · ${resultModeLabel}`
                  : `${resultModeLabel} completed`}
              </p>
            </div>
            <span className={`load-test-outcome-status ${summary.failed > 0 ? "has-failures" : "is-valid"}`}>
              {summary.failed > 0 ? `${summary.failed} failed` : "All schema-valid"}
            </span>
          </section>
          <section className="load-test-summary-grid is-compact">
            <div className="load-test-summary-card">
              <span>Achieved OTPM</span>
              <strong>{formatOtpm(summary.achieved_otpm)}</strong>
              <small>
                Total output ÷ wall time × 60 · {formatPercentage(summary.otpm_target_achievement_ratio)} of {formatOtpm(summary.target_otpm)} reference · {formatTargetSource(result.config.target_otpm_source ?? result.config.target_source)}
              </small>
            </div>
            <div className="load-test-summary-card">
              <span>Reliability</span>
              <strong>{formatPercentage(summary.succeeded / result.config.total_requests)}</strong>
              <small>{summary.succeeded} schema-valid · {summary.failed} failed · {summary.retried} retried</small>
            </div>
            <div className="load-test-summary-card">
              <span>Wall time</span>
              <strong>{formatSeconds(summary.wall_time_s)}</strong>
              <small>
                p95 request {formatSeconds(p95Latency)} · p99 {formatSeconds(summary.p99_latency_s)}
              </small>
            </div>
          </section>

          <section className="load-test-results-card">
            <Suspense
              fallback={
                <div className="load-test-chart-loading" role="status">
                  Rendering charts…
                </div>
              }
            >
              <LoadTestCharts
                result={result}
                displayName={resultDisplayName}
                theme={theme}
              />
            </Suspense>

            <div className="load-test-run-status-strip" aria-label="Retry and request status">
              <span className="load-test-retry-legend is-first-try">
                <i aria-hidden="true" /> {Math.max(summary.succeeded - summary.retried_succeeded, 0)} first try
              </span>
              <span className="load-test-retry-legend is-retried">
                <i aria-hidden="true" /> {summary.retried_succeeded} succeeded after retry
              </span>
              <span className="load-test-retry-legend is-failed">
                <i aria-hidden="true" /> {summary.failed} failed
              </span>
              <span>{summary.total_attempts} endpoint attempts</span>
            </div>

            <details className="load-test-run-details">
              <summary className="load-test-evidence-summary">
                <span>
                  <span className="load-testing-eyebrow">Run details</span>
                  <strong>Diagnostics and request evidence</strong>
                </span>
                <span className="load-testing-pill">{result.requests.length} requests</span>
              </summary>

              <div className="load-test-run-details-body">
                <section className="load-test-diagnostics" aria-label="Run diagnostics">
                  <div><span>Endpoint</span><strong>{result.model}</strong></div>
                  <div><span>Scheduling</span><strong>{result.config.scheduling_mode === "target_rate" ? "Target rate" : "Burst"}</strong></div>
                  <div><span>Requests / max in flight</span><strong>{result.config.total_requests} / {result.config.max_workers}</strong></div>
                  <div><span>Achieved ITPM</span><strong>{formatTokenRate(summary.achieved_itpm, "ITPM")}</strong></div>
                  <div><span>ITPM reference</span><strong>{formatTokenRate(summary.target_itpm, "ITPM")}</strong></div>
                  <div><span>Usage coverage</span><strong>{formatPercentage(summary.usage_coverage)}</strong></div>
                  <div><span>Peak rolling OTPM</span><strong>{formatOtpm(summary.peak_rolling_otpm)}</strong></div>
                  <div><span>Attempts</span><strong>{summary.total_attempts}</strong></div>
                  <div><span>Cache</span><strong>{result.config.cache_bust ? "Bust per request" : "Reuse allowed"}</strong></div>
                </section>

                <p className="load-test-summary-caption">
                  {`${inputTargetLabel}: ${formatTokenRate(summary.target_itpm, "ITPM")} · ${outputTargetLabel}: ${formatOtpm(summary.target_otpm)}. ${result.config.scheduling_mode === "target_rate" ? `Arrivals were clock-scheduled at ${result.config.load_multiplier ?? 1}× the limiting target.` : "References were comparison guides only; requests used burst scheduling."}`}
                </p>

                {summary.usage_missing > 0 && (
                  <p className="load-test-chart-note warning">
                    {summary.usage_missing} final response{summary.usage_missing === 1 ? "" : "s"} omitted complete usage data, so token totals may be partial.
                  </p>
                )}
                {summary.retry_exhausted > 0 && (
                  <p className="load-test-chart-note warning">
                    {summary.retry_exhausted} request{summary.retry_exhausted === 1 ? "" : "s"} exhausted all retries.
                  </p>
                )}

                <h3>Cumulative token data</h3>
                <div className="load-test-table-scroll" tabIndex={0}>
                  <table className="load-test-data-table">
                  <caption className="sr-only">
                    Cumulative input and output token totals at each completed request
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Elapsed</th>
                      <th scope="col">Input actual</th>
                      <th scope="col">Input reference</th>
                      <th scope="col">Output actual</th>
                      <th scope="col">Output reference</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.series.map((point, index) => (
                      <tr key={`chart-data-${index}`}>
                        <td>{formatSeconds(point.elapsed_s)}</td>
                        <td>{formatCompactInteger(point.cumulative_prompt_tokens)}</td>
                        <td>{formatCompactInteger(point.target_cumulative_prompt_tokens)}</td>
                        <td>{formatCompactInteger(point.cumulative_completion_tokens)}</td>
                        <td>{formatCompactInteger(point.target_cumulative_completion_tokens)}</td>
                      </tr>
                    ))}
                  </tbody>
                  </table>
                </div>

                <h3>Per-request evidence</h3>
                <p className="load-test-evidence-copy">
                  Exact start times, retries, token usage, and failures for this {resultModeLabel}.
                </p>
                <div className="load-test-table-scroll" tabIndex={0}>
                  <table className="load-test-data-table request-table">
                    <caption className="sr-only">Per-request load-test results</caption>
                    <thead>
                      <tr>
                        <th scope="col">Request</th>
                        <th scope="col">Status</th>
                        <th scope="col">Scheduled</th>
                        <th scope="col">Started</th>
                        <th scope="col">Start delay</th>
                        <th scope="col">Latency</th>
                        <th scope="col">Attempts</th>
                        <th scope="col">Retry history</th>
                        <th scope="col">Input tokens</th>
                        <th scope="col">Output tokens</th>
                        <th scope="col">Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.requests.map((request) => (
                        <tr key={request.req_id}>
                          <th scope="row">#{request.req_id + 1}</th>
                          <td>
                            <span className={`load-test-status-badge ${request.status === "ok" ? "is-ok" : "is-error"}`}>
                              {request.status}
                            </span>
                          </td>
                          <td>{formatSeconds(request.scheduled_start_s)}</td>
                          <td>{formatSeconds(request.actual_start_s)}</td>
                          <td>{formatSeconds(request.start_delay_s)}</td>
                          <td>{formatSeconds(request.elapsed_s)}</td>
                          <td>{request.attempts}</td>
                          <td className="load-test-error-cell">
                            {formatRetryHistory(request.retry_events)}
                          </td>
                          <td>{formatCompactInteger(request.prompt_tokens)}</td>
                          <td>{formatCompactInteger(request.completion_tokens)}</td>
                          <td className="load-test-error-cell">{request.error ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </details>
          </section>
        </>
      )}

      {!summary && !error && !noModels && !activeRun && !loading && (
        <section className="load-test-message-card" role="status">
          <p>
            Choose a ready endpoint, select a traffic mode, then run the test.
            References and payload details are optional advanced settings.
          </p>
        </section>
      )}

      {loading && !activeRun && (
        <section className="load-test-message-card" role="status" aria-live="polite">
          <div className="load-test-loading-row">
            <span className="spinner-lg" aria-hidden="true" />
            <div>
              <strong>Starting load test</strong>
              <p>Creating the load-test run and waiting for its first progress update.</p>
            </div>
          </div>
        </section>
      )}

      {activeRunIsLive && activeRun && (
        <>
          <section className="load-test-progress-card" role="status" aria-live="polite">
            <div className="load-test-progress-heading">
              <div>
                <span className="load-testing-eyebrow">{activeRun.status}</span>
                <strong>{progressCompleted} of {progressTotal} requests complete</strong>
                <p>{activeRun.model} · {activeRun.config?.scheduling_mode === "target_rate" ? "target rate" : "burst"} · run {activeRun.run_id.slice(0, 8)}</p>
              </div>
              <button
                type="button"
                className="btn-secondary load-test-cancel-button"
                onClick={onCancel}
                disabled={cancelling}
              >
                {cancelling ? "Cancelling…" : "Cancel run"}
              </button>
            </div>
            <div
              className="load-test-progress-track"
              role="progressbar"
              aria-label="Load test request progress"
              aria-valuemin={0}
              aria-valuemax={progressTotal || 1}
              aria-valuenow={progressCompleted}
              aria-valuetext={`${progressCompleted} of ${progressTotal} requests complete`}
            >
              <span style={{ width: `${progressPercent}%` }} />
            </div>
            <div className="load-test-progress-strip" aria-label="Current run counts">
              <span><strong>{activeRun.progress.succeeded}</strong> succeeded</span>
              <span><strong>{activeRun.progress.failed}</strong> failed</span>
              <span><strong>{activeRun.progress.total_attempts}</strong> attempts</span>
            </div>
          </section>

          {activeRun.partial_result && (
            <section className="load-test-results-card is-live">
              <Suspense
                fallback={
                  <div className="load-test-chart-loading" role="status">
                    Starting live chart…
                  </div>
                }
              >
                <LoadTestCharts
                  result={activeRun.partial_result}
                  displayName={activeRunDisplayName}
                  theme={theme}
                  live
                />
              </Suspense>
            </section>
          )}
        </>
      )}

      {activeRun?.status === "cancelled" && !summary && (
        <section className="load-test-message-card" role="status">
          <p>
            Run {activeRun.run_id.slice(0, 8)} stopped scheduling after {progressCompleted} of {progressTotal} requests.
            An endpoint call that was already in flight may still finish.
          </p>
        </section>
      )}
    </main>
  );
}

function makeSlot(selectedModel: string): ModelSlot {
  return {
    id: makeSlotId(),
    selectedModel,
    loading: false,
    result: null,
    error: null,
    notice: null,
    activeTab: "formatted",
    aiExtractPrecision: false,
    copied: false,
    outputTokenLimit: null,
    outputTokenLimitSource: null,
  };
}

function getPreferredModelName(models: ModelInfo[]): string {
  const preferredAiExtract = models.find(
    (model) =>
      model.model_type === "ai_extract" && model.availability_status === "ready"
  );
  return (
    preferredAiExtract?.name ??
    models.find((model) => model.availability_status === "ready")?.name ??
    models[0]?.name ??
    ""
  );
}

function reconcileSlotsWithModels(prevSlots: ModelSlot[], nextModels: ModelInfo[]): ModelSlot[] {
  if (nextModels.length === 0) {
    return prevSlots.map((slot) => ({ ...slot, loading: false }));
  }

  if (prevSlots.length === 0) {
    return [makeSlot(getPreferredModelName(nextModels))];
  }

  const nextModelNames = new Set(nextModels.map((model) => model.name));
  const nextModelMap = new Map(nextModels.map((model) => [model.name, model]));
  const fallbackModel = getPreferredModelName(nextModels);

  return prevSlots.map((slot) => {
    if (!nextModelNames.has(slot.selectedModel)) {
      return {
        ...slot,
        selectedModel: fallbackModel,
        loading: false,
        result: null,
        error: null,
        notice: null,
        activeTab: "formatted",
        aiExtractPrecision: false,
        copied: false,
        outputTokenLimit: null,
        outputTokenLimitSource: null,
      };
    }

    const nextModel = nextModelMap.get(slot.selectedModel);
    if (nextModel?.availability_status === "ready" && slot.notice) {
      return { ...slot, notice: null };
    }

    return slot;
  });
}

function rewriteCustomModelError(error: ModelSlot["error"], modelInfo?: ModelInfo): ModelSlot["error"] {
  if (!error || modelInfo?.model_type !== "custom") {
    return error;
  }

  if (error.error_code === "TIMEOUT") {
    return {
      ...error,
      message: CUSTOM_MODEL_TIMEOUT_MESSAGE,
      detail: error.detail || error.message,
    };
  }

  if (!CUSTOM_MODEL_WARMUP_ERROR_CODES.has(error.error_code)) {
    return error;
  }

  return {
    ...error,
    message: CUSTOM_MODEL_WARMUP_MESSAGE,
    detail: error.detail || error.message,
  };
}

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [activeView, setActiveView] = useState<AppView>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(ACTIVE_VIEW_KEY);
      if (saved === "structured-output" || saved === "load-testing") {
        return saved;
      }
    }
    return "structured-output";
  });
  const [selectedPreset, setSelectedPreset] = useState("");
  const [schemaText, setSchemaText] = useState(DEFAULT_SCHEMA);
  const [prompt, setPrompt] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(() => {
    if (typeof window !== "undefined") {
      return (localStorage.getItem(SYSTEM_PROMPT_KEY) || DEFAULT_SYSTEM_PROMPT).slice(
        0,
        SYSTEM_PROMPT_MAX_LENGTH
      );
    }
    return DEFAULT_SYSTEM_PROMPT;
  });
  const [showSystemPrompt, setShowSystemPrompt] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(SYSTEM_PROMPT_VISIBILITY_KEY) === "true";
    }
    return false;
  });
  // Empty string = use the per-endpoint default max_tokens from the backend.
  const [maxOutputTokens, setMaxOutputTokens] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(MAX_OUTPUT_TOKENS_KEY) || "";
    }
    return "";
  });
  const [schemaMode, setSchemaMode] = useState<"visual" | "json">("visual");
  const [schemaModeError, setSchemaModeError] = useState("");
  const [schemaError, setSchemaError] = useState("");
  const [schemaFocused, setSchemaFocused] = useState(false);
  const schemaFocusTitleId = useId();
  const schemaFocusRef = useRef<HTMLElement>(null);
  const schemaFocusButtonRef = useRef<HTMLButtonElement>(null);
  const [showPresetManager, setShowPresetManager] = useState(false);
  const [showModelSettings, setShowModelSettings] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [slots, setSlots] = useState<ModelSlot[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState("");
  const [presetsError, setPresetsError] = useState("");
  const [leftPaneMode, setLeftPaneMode] = useState<LeftPaneMode>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(LEFT_PANE_MODE_KEY);
      if (saved === "collapsed" || saved === "normal" || saved === "wide") {
        return saved;
      }
    }
    return "normal";
  });

  const [theme, setTheme] = useState<"light" | "dark">(() => {
    if (typeof window !== "undefined") {
      return (localStorage.getItem("theme") as "light" | "dark") || "light";
    }
    return "light";
  });
  const [loadTestModel, setLoadTestModel] = useState("");
  const [loadTestForm, setLoadTestForm] =
    useState<LoadTestFormState>(DEFAULT_LOAD_TEST_FORM);
  const [loadTestResult, setLoadTestResult] = useState<LoadTestResponse | null>(null);
  const [activeLoadTestRun, setActiveLoadTestRun] = useState<LoadTestRun | null>(null);
  const [loadTestError, setLoadTestError] = useState<ApiError | null>(null);
  const [loadTestFieldErrors, setLoadTestFieldErrors] = useState<LoadTestFieldErrors>({});
  const [loadTestLoading, setLoadTestLoading] = useState(false);
  const [loadTestCancelling, setLoadTestCancelling] = useState(false);
  const [loadTestCompletedAt, setLoadTestCompletedAt] = useState<Date | null>(null);
  const loadTestRunInFlightRef = useRef(false);
  const loadTestWorkloadInitializedRef = useRef(false);
  const [loadTestSample, setLoadTestSample] = useState<LoadTestSampleResponse | null>(null);
  const [loadTestSampleError, setLoadTestSampleError] = useState("");
  const [showLoadTestSamplePreview, setShowLoadTestSamplePreview] = useState(false);
  const [loadTestOutputEstimate, setLoadTestOutputEstimate] =
    useState<LoadTestOutputEstimateResponse | null>(null);
  const [loadTestEstimateLoading, setLoadTestEstimateLoading] = useState(false);
  const [loadTestCalibrationLoading, setLoadTestCalibrationLoading] = useState(false);
  const [loadTestEstimateError, setLoadTestEstimateError] = useState("");
  const [loadTestManualOutputEstimate, setLoadTestManualOutputEstimate] = useState(false);
  const [loadTestEstimateRefreshVersion, setLoadTestEstimateRefreshVersion] = useState(0);

  const applyLoadTestRun = useCallback((run: LoadTestRun) => {
    setActiveLoadTestRun(run);
    setLoadTestModel(run.model);
    if (run.config) {
      setLoadTestForm((current) => ({
        ...current,
        totalRequests: String(run.config?.total_requests ?? current.totalRequests),
        schedulingMode: run.config?.scheduling_mode ?? current.schedulingMode,
        durationSeconds: String(run.config?.duration_s ?? current.durationSeconds),
        loadMultiplier: String(run.config?.load_multiplier ?? current.loadMultiplier),
        expectedOutputTokens: String(
          run.config?.planned_output_tokens_per_request ?? current.expectedOutputTokens
        ),
        maxWorkers: String(run.config?.max_workers ?? current.maxWorkers),
        maxTokens: String(run.config?.max_tokens ?? current.maxTokens),
        targetItpm:
          run.config?.target_itpm == null ? "" : String(run.config.target_itpm),
        targetOtpm:
          run.config?.target_otpm == null ? "" : String(run.config.target_otpm),
      }));
    }
    const live = isLoadTestRunLive(run);
    setLoadTestLoading(live);

    if (live) {
      localStorage.setItem(ACTIVE_LOAD_TEST_RUN_KEY, run.run_id);
      setLoadTestError((current) =>
        current?.error_code === "LOAD_TEST_POLL_DELAYED" ? null : current
      );
      return;
    }

    setLoadTestCancelling(false);
    localStorage.removeItem(ACTIVE_LOAD_TEST_RUN_KEY);
    if (run.status === "completed") {
      if (run.result) {
        setLoadTestResult(run.result);
        setLoadTestEstimateRefreshVersion((current) => current + 1);
        setLoadTestCompletedAt(
          run.completed_at ? new Date(run.completed_at) : new Date()
        );
        setLoadTestError(null);
      } else {
        setLoadTestError({
          error_code: "LOAD_TEST_RESULT_MISSING",
          message: "The run completed, but its results are not available yet.",
        });
      }
      return;
    }

    if (run.status === "failed") {
      setLoadTestError(
        run.error ?? {
          error_code: "LOAD_TEST_FAILED",
          message: "The load test failed before results were produced.",
        }
      );
      return;
    }

    if (run.status === "cancelled") {
      setLoadTestError(null);
    }
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem(ACTIVE_VIEW_KEY, activeView);
  }, [activeView]);

  useEffect(() => {
    localStorage.setItem(LEFT_PANE_MODE_KEY, leftPaneMode);
  }, [leftPaneMode]);

  useEffect(() => {
    localStorage.setItem(SYSTEM_PROMPT_KEY, systemPrompt);
  }, [systemPrompt]);

  useEffect(() => {
    localStorage.setItem(SYSTEM_PROMPT_VISIBILITY_KEY, String(showSystemPrompt));
  }, [showSystemPrompt]);

  useEffect(() => {
    localStorage.setItem(MAX_OUTPUT_TOKENS_KEY, maxOutputTokens);
  }, [maxOutputTokens]);

  useEffect(() => {
    if (activeView !== "structured-output" && showPresetManager) {
      setShowPresetManager(false);
    }
  }, [activeView, showPresetManager]);

  useEffect(() => {
    if (activeView !== "structured-output" && schemaFocused) {
      setSchemaFocused(false);
    }
  }, [activeView, schemaFocused]);

  useEffect(() => {
    if (!schemaFocused) return;

    const previousBodyOverflow = document.body.style.overflow;
    const focusFrame = window.requestAnimationFrame(() => {
      schemaFocusButtonRef.current?.focus();
    });
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSchemaFocused(false);
    };

    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", handleEscape);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      document.body.style.overflow = previousBodyOverflow;
      document.removeEventListener("keydown", handleEscape);
      schemaFocusButtonRef.current?.focus();
    };
  }, [schemaFocused]);

  const refreshModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      const [availableModels, settings] = await Promise.all([
        fetchModels(),
        fetchModelSettings(),
      ]);
      const selectedNames = new Set(
        settings.endpoints.map((endpoint) => endpoint.name)
      );
      const nextModels = availableModels.filter(
        (model) => model.model_type === "ai_extract" || selectedNames.has(model.name)
      );
      setModels(nextModels);
      setSlots((prev) => reconcileSlotsWithModels(prev, nextModels));
      setModelsError("");
      return nextModels;
    } catch (error) {
      setModelsError(getErrorMessage(error, "Unable to load serving endpoints."));
      throw error;
    } finally {
      setModelsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshModels().catch((e) => console.error("Failed to load models:", e));
  }, [refreshModels]);

  const loadPresets = useCallback(() => {
    fetchPresets()
      .then((nextPresets) => {
        setPresets(nextPresets);
        setPresetsError("");
      })
      .catch((e) => {
        console.error("Failed to load presets:", e);
        setPresetsError("Presets are temporarily unavailable.");
      });
  }, []);

  useEffect(() => {
    loadPresets();
  }, [loadPresets]);

  useEffect(() => {
    fetchLoadTestSample()
      .then((sample) => {
        setLoadTestSample(sample);
        setLoadTestSampleError("");
        if (!loadTestWorkloadInitializedRef.current) {
          loadTestWorkloadInitializedRef.current = true;
          setLoadTestForm((previous) => ({
            ...previous,
            prompt: sample.prompt,
            schemaJson: sample.schema_json,
            systemPrompt: sample.system_prompt,
          }));
        }
      })
      .catch((e) => {
        console.error("Failed to load benchmark sample:", e);
        setLoadTestSampleError("Unable to load the benchmark sample preview.");
      });
  }, []);

  useEffect(() => {
    let ignore = false;
    const recoverRun = async () => {
      const rememberedRunId = localStorage.getItem(ACTIVE_LOAD_TEST_RUN_KEY);
      try {
        const runs = await fetchActiveLoadTestRuns();
        if (ignore) return;
        if (runs.length > 0) {
          setLoadTestResult(null);
          setLoadTestCompletedAt(null);
          applyLoadTestRun(runs[0]);
          return;
        }

        if (rememberedRunId) {
          const rememberedRun = await fetchLoadTestRun(rememberedRunId);
          if (!ignore) applyLoadTestRun(rememberedRun);
        }
      } catch (error) {
        if (ignore) return;
        if (rememberedRunId) {
          localStorage.removeItem(ACTIVE_LOAD_TEST_RUN_KEY);
          setLoadTestLoading(false);
          setLoadTestError({
            error_code: "LOAD_TEST_RUN_EXPIRED",
            message: "The previous run can no longer be recovered.",
            detail: "It may have expired or the app may have restarted. Start a new run when ready.",
          });
          return;
        }
        console.warn("Unable to recover an active load-test run:", error);
      }
    };

    recoverRun()
      .catch((error) => {
        console.warn("Unable to recover an active load-test run:", error);
      });
    return () => {
      ignore = true;
    };
  }, [applyLoadTestRun]);

  useEffect(() => {
    const runId = activeLoadTestRun?.run_id;
    if (!runId || !isLoadTestRunLive(activeLoadTestRun)) return;

    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const nextRun = await fetchLoadTestRun(runId);
        if (stopped) return;
        applyLoadTestRun(nextRun);
        if (isLoadTestRunLive(nextRun)) {
          timer = setTimeout(poll, 1000);
        }
      } catch (error) {
        if (stopped) return;
        setLoadTestError({
          error_code: "LOAD_TEST_POLL_DELAYED",
          message: "Progress updates were briefly interrupted. Retrying automatically.",
          detail: getErrorMessage(error, "Unable to reach the run status endpoint."),
        });
        timer = setTimeout(poll, 1500);
      }
    };

    timer = setTimeout(poll, 500);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [activeLoadTestRun?.run_id, activeLoadTestRun?.status, applyLoadTestRun]);

  const loadTestModels = useMemo(
    () =>
      models.filter(
        (model) =>
          model.model_type !== "ai_extract" && model.availability_status === "ready"
      ),
    [models]
  );
  const selectedLoadTestModelInfo =
    loadTestModels.find((model) => model.name === loadTestModel) ?? null;

  const loadTestEstimateRequest = useMemo<LoadTestOutputEstimateRequest | null>(() => {
    const maxTokens = parseIntegerInput(loadTestForm.maxTokens);
    if (
      loadTestForm.schedulingMode !== "target_rate" ||
      !loadTestForm.targetOtpm.trim() ||
      !loadTestModel ||
      maxTokens == null ||
      !loadTestForm.prompt.trim() ||
      !loadTestForm.schemaJson.trim()
    ) {
      return null;
    }
    return {
      model: loadTestModel,
      max_tokens: maxTokens,
      prompt: loadTestForm.prompt,
      schema_json: loadTestForm.schemaJson,
      system_prompt: loadTestForm.systemPrompt,
    };
  }, [
    loadTestForm.maxTokens,
    loadTestForm.prompt,
    loadTestForm.schemaJson,
    loadTestForm.schedulingMode,
    loadTestForm.systemPrompt,
    loadTestForm.targetOtpm,
    loadTestModel,
  ]);

  useEffect(() => {
    if (!loadTestEstimateRequest || loadTestManualOutputEstimate) {
      setLoadTestEstimateLoading(false);
      if (!loadTestEstimateRequest) {
        setLoadTestOutputEstimate(null);
        setLoadTestEstimateError("");
      }
      return;
    }

    let ignore = false;
    setLoadTestEstimateLoading(true);
    setLoadTestEstimateError("");
    setLoadTestForm((current) => ({
      ...current,
      expectedOutputTokens: "",
    }));
    const timer = setTimeout(() => {
      fetchLoadTestOutputEstimate(loadTestEstimateRequest)
        .then((estimate) => {
          if (ignore) return;
          setLoadTestOutputEstimate(estimate);
          setLoadTestForm((current) => ({
            ...current,
            expectedOutputTokens:
              estimate.tokens == null ? "" : String(estimate.tokens),
          }));
        })
        .catch((error) => {
          if (ignore) return;
          setLoadTestOutputEstimate(null);
          setLoadTestForm((current) => ({
            ...current,
            expectedOutputTokens: "",
          }));
          setLoadTestEstimateError(
            getErrorMessage(error, "Unable to load the saved output estimate.")
          );
        })
        .finally(() => {
          if (!ignore) setLoadTestEstimateLoading(false);
        });
    }, 350);

    return () => {
      ignore = true;
      clearTimeout(timer);
    };
  }, [
    loadTestEstimateRequest,
    loadTestEstimateRefreshVersion,
    loadTestManualOutputEstimate,
  ]);

  const discardTerminalLoadTest = useCallback(() => {
    setActiveLoadTestRun((current) => (isLoadTestRunLive(current) ? current : null));
    setLoadTestResult(null);
    setLoadTestCompletedAt(null);
    setLoadTestError(null);
  }, []);

  useEffect(() => {
    if (loadTestModels.length === 0) {
      if (loadTestModel && !isLoadTestRunLive(activeLoadTestRun)) {
        setLoadTestModel("");
      }
      return;
    }

    if (
      !isLoadTestRunLive(activeLoadTestRun) &&
      !loadTestModels.some((model) => model.name === loadTestModel)
    ) {
      const nextModel = loadTestModels[0];
      setLoadTestModel(nextModel.name);
      setLoadTestForm((previous) => ({
        ...previous,
        maxTokens: String(
          Math.min(
            DEFAULT_LOAD_TEST_CONFIG.max_tokens,
            nextModel.max_tokens ?? LOAD_TEST_MAX_TOKENS,
            LOAD_TEST_MAX_TOKENS
          )
        ),
        expectedOutputTokens:
          parseIntegerInput(previous.expectedOutputTokens) == null
            ? ""
            : String(
                Math.min(
                  parseIntegerInput(previous.expectedOutputTokens)!,
                  nextModel.max_tokens ?? LOAD_TEST_MAX_TOKENS,
                  LOAD_TEST_MAX_TOKENS
                )
              ),
      }));
    }
  }, [activeLoadTestRun, loadTestModels, loadTestModel]);

  const updateSlot = useCallback((id: string, patch: Partial<ModelSlot>) => {
    setSlots((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  }, []);

  const handleLoadTestFormChange = useCallback(
    (field: keyof LoadTestFormState, value: string) => {
      if (field === "prompt" || field === "schemaJson" || field === "systemPrompt") {
        loadTestWorkloadInitializedRef.current = true;
      }
      setLoadTestForm((prev) => ({ ...prev, [field]: value }));
      setLoadTestFieldErrors((previous) => {
        if (!previous[field]) return previous;
        const next = { ...previous };
        delete next[field];
        return next;
      });
      discardTerminalLoadTest();
    },
    [discardTerminalLoadTest]
  );

  const handleLoadTestModelChange = useCallback(
    (modelName: string) => {
      const nextModel = loadTestModels.find((model) => model.name === modelName);
      setLoadTestModel(modelName);
      setLoadTestForm((previous) => ({
        ...previous,
        maxTokens: String(
          Math.min(
            parseIntegerInput(previous.maxTokens) ?? DEFAULT_LOAD_TEST_CONFIG.max_tokens,
            nextModel?.max_tokens ?? LOAD_TEST_MAX_TOKENS,
            LOAD_TEST_MAX_TOKENS
          )
        ),
        expectedOutputTokens:
          parseIntegerInput(previous.expectedOutputTokens) == null
            ? ""
            : String(
                Math.min(
                  parseIntegerInput(previous.expectedOutputTokens)!,
                  nextModel?.max_tokens ?? LOAD_TEST_MAX_TOKENS,
                  LOAD_TEST_MAX_TOKENS
                )
              ),
      }));
      setLoadTestFieldErrors({});
      discardTerminalLoadTest();
    },
    [discardTerminalLoadTest, loadTestModels]
  );

  const handleCalibrateLoadTestOutput = useCallback(async () => {
    if (!loadTestEstimateRequest || loadTestCalibrationLoading) {
      if (!loadTestEstimateRequest) {
        setLoadTestEstimateError("Choose an endpoint and complete the workload before calibrating.");
      }
      return;
    }
    setLoadTestCalibrationLoading(true);
    setLoadTestEstimateError("");
    try {
      const estimate = await calibrateLoadTestOutputEstimate(loadTestEstimateRequest);
      setLoadTestOutputEstimate(estimate);
      setLoadTestForm((current) => ({
        ...current,
        expectedOutputTokens: estimate.tokens == null ? "" : String(estimate.tokens),
      }));
      setLoadTestFieldErrors((current) => ({
        ...current,
        expectedOutputTokens: undefined,
      }));
      discardTerminalLoadTest();
    } catch (error) {
      setLoadTestEstimateError(
        getErrorMessage(error, "Unable to calibrate output size.")
      );
    } finally {
      setLoadTestCalibrationLoading(false);
    }
  }, [
    discardTerminalLoadTest,
    loadTestCalibrationLoading,
    loadTestEstimateRequest,
  ]);

  const handleManualOutputEstimateChange = useCallback(
    (manual: boolean) => {
      setLoadTestManualOutputEstimate(manual);
      setLoadTestForm((current) => ({
        ...current,
        expectedOutputTokens: manual
          ? current.expectedOutputTokens
          : loadTestOutputEstimate?.tokens == null
            ? ""
            : String(loadTestOutputEstimate.tokens),
      }));
      setLoadTestFieldErrors((current) => ({
        ...current,
        expectedOutputTokens: undefined,
      }));
      setLoadTestEstimateError("");
      discardTerminalLoadTest();
    },
    [discardTerminalLoadTest, loadTestOutputEstimate]
  );

  const handleResetLoadTestTarget = useCallback(() => {
    setLoadTestForm((prev) => ({
      ...prev,
      targetItpm: String(DEFAULT_REFERENCE_ITPM),
      targetOtpm: String(DEFAULT_REFERENCE_OTPM),
    }));
    setLoadTestFieldErrors((previous) => ({
      ...previous,
      targetItpm: undefined,
      targetOtpm: undefined,
    }));
    discardTerminalLoadTest();
  }, [discardTerminalLoadTest]);

  const handleRestoreLoadTestSample = useCallback(() => {
    if (!loadTestSample) return;
    loadTestWorkloadInitializedRef.current = true;
    setLoadTestForm((previous) => ({
      ...previous,
      prompt: loadTestSample.prompt,
      schemaJson: loadTestSample.schema_json,
      systemPrompt: loadTestSample.system_prompt,
    }));
    setLoadTestFieldErrors({});
    discardTerminalLoadTest();
  }, [discardTerminalLoadTest, loadTestSample]);

  const handleUsePlaygroundForLoadTest = useCallback(() => {
    loadTestWorkloadInitializedRef.current = true;
    setLoadTestForm((previous) => ({
      ...previous,
      prompt,
      schemaJson: schemaText,
      systemPrompt,
    }));
    setLoadTestFieldErrors({});
    discardTerminalLoadTest();
    setShowLoadTestSamplePreview(true);
  }, [discardTerminalLoadTest, prompt, schemaText, systemPrompt]);

  const handlePresetChange = useCallback(
    (id: string) => {
      setSelectedPreset(id);
      setSlots((prev) =>
        prev.map((s) => ({
          ...s,
          result: null,
          error: null,
          notice: null,
          activeTab: "formatted" as const,
          outputTokenLimit: null,
          outputTokenLimitSource: null,
        }))
      );
      const preset = presets.find((p) => p.id === id);
      if (preset) {
        setSchemaText(preset.schema_json);
        setPrompt(preset.example_prompt);
        setSystemPrompt(preset.system_prompt);
        setSchemaError("");
      }
    },
    [presets]
  );

  const validateSchema = (): boolean => {
    try {
      const parsed = JSON.parse(schemaText);
      if (!parsed || typeof parsed !== "object" || !("type" in parsed)) {
        setSchemaError('Schema must have a "type" key');
        return false;
      }
      setSchemaError("");
      return true;
    } catch {
      setSchemaError("Invalid JSON in schema");
      return false;
    }
  };

  const anyLoading = slots.some((s) => s.loading);

  const handleSubmit = async () => {
    if (!validateSchema()) return;

    // Resolve the optional per-request output cap. Empty = use the endpoint
    // default; a value must be a positive integer within the allowed ceiling.
    let maxOutputTokensOverride: number | null = null;
    const maxOutputTokensText = maxOutputTokens.trim();
    if (maxOutputTokensText.length > 0) {
      const parsed = parseIntegerInput(maxOutputTokensText);
      if (parsed == null || parsed < 1 || parsed > MAX_OUTPUT_TOKENS_CEILING) {
        setSchemaError(
          `Max output tokens must be between 1 and ${MAX_OUTPUT_TOKENS_CEILING}, or left blank for the model default.`
        );
        return;
      }
      maxOutputTokensOverride = parsed;
    }

    // Surface progress immediately while the endpoint list is refreshed. Without
    // this preflight state, slower workspaces appear to ignore the first click.
    setSlots((prev) =>
      prev.map((slot) => ({
        ...slot,
        loading: true,
        result: null,
        error: null,
        notice: null,
        copied: false,
        outputTokenLimit: null,
        outputTokenLimitSource: null,
      }))
    );

    let latestModels = models;
    try {
      latestModels = await refreshModels();
    } catch (e) {
      console.error("Failed to refresh models before inference:", e);
    }

    const currentSlots = reconcileSlotsWithModels(slots, latestModels);
    const latestModelsByName = new Map(latestModels.map((model) => [model.name, model]));

    if (latestModels.length === 0) {
      setSlots(
        currentSlots.map((slot) => ({
          ...slot,
          loading: false,
          result: null,
          error: null,
          notice: NO_MODELS_AVAILABLE_MESSAGE,
          activeTab: "formatted" as const,
          copied: false,
          outputTokenLimit: null,
          outputTokenLimitSource: null,
        }))
      );
      return;
    }

    setSlots(
      currentSlots.map((slot) => {
        const selectedModelInfo = latestModelsByName.get(slot.selectedModel);
        const isUnavailableModel =
          !selectedModelInfo || selectedModelInfo.availability_status !== "ready";

        if (isUnavailableModel) {
          return {
            ...slot,
            loading: false,
            result: null,
            error: null,
            notice:
              selectedModelInfo?.availability_message || MODEL_UNAVAILABLE_MESSAGE,
            activeTab: "formatted" as const,
            copied: false,
            outputTokenLimit: null,
            outputTokenLimitSource: null,
          };
        }

        return {
          ...slot,
          loading: true,
          result: null,
          error: null,
          notice: null,
          activeTab: "formatted" as const,
          copied: false,
          outputTokenLimit:
            selectedModelInfo?.model_type === "ai_extract"
              ? null
              : maxOutputTokensOverride ?? selectedModelInfo?.max_tokens ?? null,
          outputTokenLimitSource:
            selectedModelInfo?.model_type === "ai_extract"
              ? null
              : maxOutputTokensOverride == null
                ? "model_default" as const
                : "override" as const,
        };
      })
    );

    const runnableSlots = currentSlots.filter((slot) => {
      const selectedModelInfo = latestModelsByName.get(slot.selectedModel);
      return selectedModelInfo?.availability_status === "ready";
    });

    await Promise.allSettled(
      runnableSlots.map(async (slot) => {
        const selectedModelInfo = latestModelsByName.get(slot.selectedModel);
        try {
          const res = await runInference({
            prompt,
            schema_json: schemaText,
            system_prompt: systemPrompt,
            model: slot.selectedModel,
            runtime:
              selectedModelInfo?.model_type === "ai_extract"
                ? "ai_extract"
                : "model_serving",
            ai_extract_options:
              selectedModelInfo?.model_type === "ai_extract"
                ? { precision: slot.aiExtractPrecision }
                : null,
            max_tokens:
              selectedModelInfo?.model_type === "ai_extract"
                ? null
                : maxOutputTokensOverride,
          });
          updateSlot(slot.id, {
            result: res,
            loading: false,
            error: null,
            notice: null,
            activeTab: "formatted",
          });
        } catch (e: unknown) {
          let nextError: ModelSlot["error"];
          if (e && typeof e === "object" && "error_code" in e) {
            nextError = rewriteCustomModelError(
              e as ModelSlot["error"],
              selectedModelInfo
            );
          } else {
            nextError = {
              error_code: "UNKNOWN",
              message: e instanceof Error ? e.message : "Unknown error",
            };
          }

          updateSlot(slot.id, {
            error: nextError,
            loading: false,
            notice: null,
          });
        }
      })
    );
  };

  const toLoadTestApiError = (error: unknown): ApiError =>
    error && typeof error === "object" && "error_code" in error
      ? (error as ApiError)
      : {
          error_code: "UNKNOWN",
          message: error instanceof Error ? error.message : "Unknown error",
        };

  const validateLoadTestDraft = (modelInfo = selectedLoadTestModelInfo) => {
    const validation = buildLoadTestRequest(loadTestForm, modelInfo);
    setLoadTestFieldErrors(validation.errors);
    if (!validation.request) {
      setLoadTestError(null);
      if (validation.errors.prompt || validation.errors.schemaJson || validation.errors.systemPrompt) {
        setShowLoadTestSamplePreview(true);
      }
    }
    return validation.request;
  };

  const handleRunLoadTest = async () => {
    if (loadTestRunInFlightRef.current || isLoadTestRunLive(activeLoadTestRun)) return;
    loadTestRunInFlightRef.current = true;
    setLoadTestLoading(true);
    setLoadTestError(null);
    let runStarted = false;

    try {
      const request = validateLoadTestDraft(selectedLoadTestModelInfo);
      if (!request) return;

      setActiveLoadTestRun(null);
      setLoadTestResult(null);
      setLoadTestCompletedAt(null);
      const nextRun = await startLoadTest(request);
      runStarted = true;
      applyLoadTestRun(nextRun);
    } catch (error: unknown) {
      const apiError = toLoadTestApiError(error);
      if (apiError.error_code === "LOAD_TEST_RUN_ALREADY_ACTIVE") {
        try {
          const runs = await fetchActiveLoadTestRuns();
          if (runs.length > 0) {
            runStarted = true;
            applyLoadTestRun(runs[0]);
            return;
          }
        } catch (recoveryError) {
          console.warn("Unable to recover the existing load-test run:", recoveryError);
        }
      }
      setLoadTestError(apiError);
    } finally {
      loadTestRunInFlightRef.current = false;
      if (!runStarted) setLoadTestLoading(false);
    }
  };

  const handleCancelLoadTest = async () => {
    if (!activeLoadTestRun || !isLoadTestRunLive(activeLoadTestRun) || loadTestCancelling) {
      return;
    }

    setLoadTestCancelling(true);
    setLoadTestError(null);
    try {
      const nextRun = await cancelLoadTestRun(activeLoadTestRun.run_id);
      applyLoadTestRun(nextRun);
    } catch (error: unknown) {
      setLoadTestCancelling(false);
      setLoadTestError(toLoadTestApiError(error));
    }
  };

  const addSlot = () => {
    if (slots.length >= 3 || models.length === 0) return;
    setSlots((prev) => [...prev, makeSlot(getPreferredModelName(models))]);
  };

  const removeSlot = (id: string) => {
    setSlots((prev) => (prev.length <= 1 ? prev : prev.filter((s) => s.id !== id)));
  };

  const handleSchemaFocusKeyDown = (event: ReactKeyboardEvent<HTMLElement>) => {
    if (!schemaFocused || event.key !== "Tab") return;
    const focusable = Array.from(
      schemaFocusRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      ) ?? []
    ).filter((element) => element.getClientRects().length > 0);
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const promptLen = prompt.length;
  const systemPromptLen = systemPrompt.length;
  const systemPromptDefault =
    presets.find((preset) => preset.id === selectedPreset)?.system_prompt ??
    DEFAULT_SYSTEM_PROMPT;
  const showLoadTestTargetReset =
    loadTestForm.targetItpm.trim() !== String(DEFAULT_REFERENCE_ITPM) ||
    loadTestForm.targetOtpm.trim() !== String(DEFAULT_REFERENCE_OTPM);
  const paneToggle = (
    <div className="pane-toggle" role="group" aria-label="Left pane size">
      <div className="pane-toggle-group">
        <button
          className={`pane-toggle-btn ${leftPaneMode === "collapsed" ? "active" : ""}`}
          onClick={() => setLeftPaneMode("collapsed")}
          aria-pressed={leftPaneMode === "collapsed"}
        >
          Hide
        </button>
        <button
          className={`pane-toggle-btn ${leftPaneMode === "normal" ? "active" : ""}`}
          onClick={() => setLeftPaneMode("normal")}
          aria-pressed={leftPaneMode === "normal"}
        >
          Edit
        </button>
        <button
          className={`pane-toggle-btn ${leftPaneMode === "wide" ? "active" : ""}`}
          onClick={() => setLeftPaneMode("wide")}
          aria-pressed={leftPaneMode === "wide"}
        >
          Wide
        </button>
      </div>
    </div>
  );

  const structuredOutputView = (
    <main
      id="main-workspace"
      className={`main pane-${leftPaneMode}`}
      aria-busy={anyLoading}
    >
      <div className={`panel panel-left ${leftPaneMode === "collapsed" ? "is-collapsed" : ""}`}>
        <section
          ref={schemaFocusRef}
          className={`schema-workspace ${schemaFocused ? "is-focused" : ""}`}
          role={schemaFocused ? "dialog" : undefined}
          aria-modal={schemaFocused ? "true" : undefined}
          aria-labelledby={schemaFocusTitleId}
          onKeyDown={handleSchemaFocusKeyDown}
        >
        <div className="schema-header">
          <div className="section-heading-copy">
            <span className="section-kicker">1 · Output contract</span>
            <h2 id={schemaFocusTitleId}>Define the schema</h2>
            <p>Required controls property presence; Nullable allows a field value to be null.</p>
          </div>
          <div className="schema-header-actions">
            <div className="schema-mode-toggle" role="group" aria-label="Schema editor mode">
              <button
                id="schema-tab-visual"
                className={`tab ${schemaMode === "visual" ? "active" : ""}`}
                aria-pressed={schemaMode === "visual"}
                disabled={anyLoading}
                onClick={() => {
                  try {
                    JSON.parse(schemaText);
                    setSchemaModeError("");
                    setSchemaMode("visual");
                  } catch {
                    setSchemaModeError("Fix JSON errors before switching to Visual mode");
                  }
                }}
              >
                Visual
              </button>
              <button
                id="schema-tab-json"
                className={`tab ${schemaMode === "json" ? "active" : ""}`}
                aria-pressed={schemaMode === "json"}
                disabled={anyLoading}
                onClick={() => {
                  setSchemaModeError("");
                  setSchemaMode("json");
                }}
              >
                JSON
              </button>
            </div>
            <button
              ref={schemaFocusButtonRef}
              type="button"
              className={`btn-inline-secondary schema-focus-button ${schemaFocused ? "active" : ""}`}
              aria-pressed={schemaFocused}
              aria-label={schemaFocused ? "Exit schema focus mode" : "Open schema focus mode"}
              onClick={() => setSchemaFocused((current) => !current)}
              disabled={anyLoading}
            >
              <FocusIcon />
              {schemaFocused ? "Done" : "Focus"}
            </button>
          </div>
        </div>
        {schemaModeError && <div className="sb-parse-error" role="alert">{schemaModeError}</div>}
        {schemaError && <div className="schema-error" role="alert">{schemaError}</div>}
        <div
          id={`schema-panel-${schemaMode}`}
          className={`schema-editor ${schemaMode === "json" ? "json-mode" : "visual-mode"}`}
          aria-label="JSON schema editor"
        >
          {schemaMode === "json" ? (
            <Editor
              language="json"
              theme={theme === "dark" ? "vs-dark" : "vs"}
              value={schemaText}
              onChange={(v) => {
                setSchemaText(v ?? "{}");
                setSchemaError("");
              }}
              options={{
                minimap: { enabled: false },
                fontSize: 13,
                lineNumbers: "on",
                readOnly: anyLoading,
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 2,
                wordWrap: "on",
                renderLineHighlight: "gutter",
                cursorBlinking: "smooth",
                lineNumbersMinChars: 3,
                overviewRulerBorder: false,
              }}
            />
          ) : (
            <SchemaBuilder schemaText={schemaText} onChange={setSchemaText} disabled={anyLoading} />
          )}
        </div>
        </section>

        <div className="prompt-toolbar">
          <div className="section-heading-copy compact">
            <span className="section-kicker">2 · Source</span>
            <label htmlFor="medical-text">Medical text</label>
          </div>
          <button
            type="button"
            className={`btn-inline-secondary ${showSystemPrompt ? "active" : ""}`}
            onClick={() => setShowSystemPrompt((prev) => !prev)}
            aria-expanded={showSystemPrompt}
            aria-controls="system-prompt-editor"
            disabled={anyLoading}
          >
            {showSystemPrompt ? "Hide System Prompt" : "Show System Prompt"}
          </button>
        </div>
        <div className="system-prompt-meta">
          {selectedPreset
            ? "Tailored to the selected preset and shared by every comparison model."
            : "The system prompt is shared by every comparison model."}
        </div>
        {showSystemPrompt && (
          <div className="system-prompt-panel" id="system-prompt-editor">
            <div className="system-prompt-header">
              <label htmlFor="system-prompt-input">System Prompt</label>
              <button
                type="button"
                className="btn-inline-secondary"
                onClick={() => setSystemPrompt(systemPromptDefault)}
                disabled={anyLoading || systemPrompt === systemPromptDefault}
              >
                Reset
              </button>
            </div>
            <textarea
              id="system-prompt-input"
              name="system-prompt"
              className="system-prompt-input"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              maxLength={SYSTEM_PROMPT_MAX_LENGTH}
              placeholder="Add high-level extraction instructions..."
              aria-label="System prompt input"
              disabled={anyLoading}
            />
            <div
              className={`char-count system-prompt-count ${
                systemPromptLen > SYSTEM_PROMPT_MAX_LENGTH - 200 ? "warning" : ""
              }`}
            >
              {systemPromptLen.toLocaleString()} / {SYSTEM_PROMPT_MAX_LENGTH.toLocaleString()}
            </div>
          </div>
        )}

        <textarea
          id="medical-text"
          name="medical-text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          maxLength={MEDICAL_TEXT_MAX_LENGTH}
          placeholder="Paste or type a clinical note..."
          aria-label="Medical text input"
          disabled={anyLoading}
        />
        <div
          className={`char-count ${
            promptLen > MEDICAL_TEXT_MAX_LENGTH - 1000 ? "warning" : ""
          }`}
        >
          {promptLen.toLocaleString()} / {MEDICAL_TEXT_MAX_LENGTH.toLocaleString()}
        </div>

        <div className="run-settings">
          <div className="max-tokens-field">
            <label htmlFor="max-output-tokens">Output limit</label>
            <input
              id="max-output-tokens"
              name="max-output-tokens"
              type="number"
              min={1}
              max={MAX_OUTPUT_TOKENS_CEILING}
              step={1}
              value={maxOutputTokens}
              placeholder="Model default"
              onChange={(e) => setMaxOutputTokens(e.target.value)}
              aria-label="Max output tokens override"
              disabled={anyLoading}
            />
            <span className="max-tokens-hint">
              Serving models only · blank uses each model&apos;s configured default · override up to{" "}
              {MAX_OUTPUT_TOKENS_CEILING.toLocaleString()}
            </span>
          </div>

          <button
            className="btn-submit"
            onClick={handleSubmit}
            disabled={anyLoading || modelsLoading || slots.length === 0 || !prompt.trim()}
            aria-busy={anyLoading}
          >
            {anyLoading && <span className="spinner" />}
            {anyLoading ? "Extracting…" : "Run extraction"}
          </button>
        </div>
      </div>

      <div className="panel-right-container">
        {slots.length === 0 && (
          <section
            className="model-empty-state"
            role={modelsError ? "alert" : "status"}
            aria-live="polite"
          >
            <div className="model-empty-icon" aria-hidden="true">{"{}"}</div>
            <span className="section-kicker">3 · Compare output</span>
            <h2>
              {modelsLoading
                ? "Loading serving endpoints"
                : modelsError
                  ? "Models could not be loaded"
                  : "Choose models for this workspace"}
            </h2>
            <p>
              {modelsLoading
                ? "Checking the endpoints available to this app…"
                : modelsError || "Select at least one endpoint, then run an extraction to review structured JSON."}
            </p>
            {!modelsLoading && (
              <div className="model-empty-actions">
                {modelsError && (
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() => {
                      refreshModels().catch((e) => console.error("Failed to refresh models:", e));
                    }}
                  >
                    Retry
                  </button>
                )}
                <button
                  type="button"
                  className="btn-submit model-empty-primary"
                  onClick={() => setShowModelSettings(true)}
                >
                  Configure models
                </button>
              </div>
            )}
          </section>
        )}
        {slots.map((slot) => (
          <ModelResultColumn
            key={slot.id}
            slot={slot}
            models={models}
            sourceText={prompt}
            onModelChange={(model) =>
              updateSlot(slot.id, {
                selectedModel: model,
                result: null,
                notice: null,
                error: null,
                activeTab: "formatted",
                aiExtractPrecision: false,
                copied: false,
                outputTokenLimit: null,
                outputTokenLimitSource: null,
              })
            }
            onRemove={() => removeSlot(slot.id)}
            onUpdateSlot={(patch) => updateSlot(slot.id, patch)}
            onRefreshModels={() => {
              refreshModels().catch((e) => console.error("Failed to refresh models:", e));
            }}
            canRemove={slots.length > 1 && !anyLoading}
          />
        ))}
        {slots.length > 0 && models.length > 1 && slots.length < 3 && (
          <button
            className="btn-add-slot"
            onClick={addSlot}
            aria-label="Compare models: add another model"
            disabled={anyLoading}
          >
            <span className="add-slot-icon" aria-hidden="true">+</span>
            <span className="add-slot-label">Compare</span>
          </button>
        )}
      </div>
    </main>
  );

  return (
    <div className="app">
      <a className="skip-link" href="#main-workspace">Skip to workspace</a>
      <header className="app-header">
        <div className="topbar">
          <div className="topbar-brand">
            <h1>Clinical Extractor</h1>
            <p>Schema-guided extraction · Databricks Model Serving</p>
          </div>

          <nav className="topbar-nav" aria-label="Application views">
            {APP_VIEW_OPTIONS.map((view) => (
              <button
                key={view.id}
                type="button"
                className={`topbar-nav-button ${activeView === view.id ? "active" : ""}`}
                aria-current={activeView === view.id ? "page" : undefined}
                onClick={() => setActiveView(view.id)}
              >
                {view.title}
              </button>
            ))}
          </nav>

          <div className="topbar-utilities" aria-label="Application utilities">
            <button
              type="button"
              className="topbar-icon-button"
              aria-label="Help"
              title="Help"
              onClick={() => {
                setShowHelp(true);
                setShowModelSettings(false);
              }}
            >
              <HelpIcon />
            </button>
            <button
              type="button"
              className="topbar-icon-button"
              aria-label="Settings"
              title="Settings"
              onClick={() => {
                setShowModelSettings(true);
                setShowHelp(false);
              }}
              disabled={anyLoading}
            >
              <SettingsIcon />
            </button>
            <button
              type="button"
              className="topbar-icon-button"
              onClick={() => setTheme(theme === "light" ? "dark" : "light")}
              aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
              title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            >
              <ThemeIcon theme={theme} />
            </button>
          </div>
        </div>

        {activeView === "structured-output" && (
          <div className="context-toolbar" aria-label="Playground controls">
            <div className="context-toolbar-group context-toolbar-editor">
              <span className="context-toolbar-label">Editor</span>
              {paneToggle}
            </div>

            <div className="context-toolbar-group context-toolbar-presets">
              <span className="context-toolbar-label">Preset</span>
              <div className="context-preset-controls">
                <select
                  id="preset-select"
                  name="preset-select"
                  className="context-preset-select"
                  value={selectedPreset}
                  onChange={(e) => handlePresetChange(e.target.value)}
                  aria-label="Select a preset"
                  disabled={anyLoading || (Boolean(presetsError) && presets.length === 0)}
                >
                  <option value="">
                    {presetsError && presets.length === 0
                      ? "Presets unavailable"
                      : presetsError
                        ? "Select a cached preset…"
                        : "Select a preset…"}
                  </option>
                  {presets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="context-toolbar-action"
                  onClick={() => setShowPresetManager(true)}
                  disabled={anyLoading}
                >
                  Manage
                </button>
              </div>
              {presetsError && (
                <div className="context-preset-warning">
                  <span role="status">
                    {presets.length > 0 ? "Preset refresh failed." : presetsError}
                  </span>
                  <button type="button" onClick={loadPresets} disabled={anyLoading}>
                    Retry
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </header>

      {activeView === "structured-output" ? (
        structuredOutputView
      ) : (
        <LoadTestingView
          theme={theme}
          models={loadTestModels}
          selectedModelInfo={selectedLoadTestModelInfo}
          selectedModel={loadTestModel}
          onModelChange={handleLoadTestModelChange}
          form={loadTestForm}
          onFormChange={handleLoadTestFormChange}
          onResetTarget={handleResetLoadTestTarget}
          showTargetReset={showLoadTestTargetReset}
          onRun={handleRunLoadTest}
          onCancel={handleCancelLoadTest}
          onRestoreSample={handleRestoreLoadTestSample}
          onUsePlayground={handleUsePlaygroundForLoadTest}
          outputEstimate={loadTestOutputEstimate}
          estimateLoading={loadTestEstimateLoading}
          calibrationLoading={loadTestCalibrationLoading}
          estimateError={loadTestEstimateError}
          manualOutputEstimate={loadTestManualOutputEstimate}
          onCalibrateOutput={handleCalibrateLoadTestOutput}
          onManualOutputEstimateChange={handleManualOutputEstimateChange}
          loading={loadTestLoading}
          cancelling={loadTestCancelling}
          activeRun={activeLoadTestRun}
          result={loadTestResult}
          error={loadTestError}
          fieldErrors={loadTestFieldErrors}
          completedAt={loadTestCompletedAt}
          sample={loadTestSample}
          sampleError={loadTestSampleError}
          showSamplePreview={showLoadTestSamplePreview}
          onToggleSamplePreview={() =>
            setShowLoadTestSamplePreview((prev) => !prev)
          }
        />
      )}

      {showPresetManager && activeView === "structured-output" && (
        <PresetManager
          presets={presets}
          currentSchema={schemaText}
          currentPrompt={prompt}
          currentSystemPrompt={systemPrompt}
          onClose={() => setShowPresetManager(false)}
          onPresetsChanged={loadPresets}
        />
      )}

      {showModelSettings && (
        <ModelSettingsManager
          onClose={() => setShowModelSettings(false)}
          onModelsChanged={async () => {
            try {
              await refreshModels();
            } catch (e) {
              console.error("Failed to refresh models after settings change:", e);
              throw e;
            }
          }}
        />
      )}

      {showHelp && <HelpDrawer onClose={() => setShowHelp(false)} />}
    </div>
  );
}

export default App;
