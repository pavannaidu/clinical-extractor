import { useEffect, useRef, useState } from "react";
import type {
  ExtractionCitation,
  ExtractionEvidence,
  ModelInfo,
  ModelSlot,
} from "./types";
import FormattedView from "./FormattedView";
import {
  MAX_OUTPUT_TOKENS_CEILING,
  validateOutputTokenOverride,
} from "./outputTokens";

interface Props {
  slot: ModelSlot;
  models: ModelInfo[];
  sourceText: string;
  onModelChange: (model: string) => void;
  onRemove: () => void;
  onUpdateSlot: (patch: Partial<ModelSlot>) => void;
  onRefreshModels: () => void;
  canRemove: boolean;
}

function displayModelName(model: ModelInfo | undefined, fallback: string): string {
  return model?.display_name || model?.name || fallback;
}

function formatEvidenceValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null) return "null";
  if (value === undefined) return "—";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function EvidenceSnippet({
  citation,
  sourceText,
}: {
  citation: ExtractionCitation;
  sourceText: string;
}) {
  if (
    typeof citation.start !== "number" ||
    typeof citation.stop !== "number" ||
    citation.start < 0 ||
    citation.stop <= citation.start ||
    citation.start >= sourceText.length
  ) {
    return null;
  }
  const start = Math.min(citation.start, sourceText.length);
  const stop = Math.min(citation.stop, sourceText.length);
  const contextStart = Math.max(0, start - 36);
  const contextStop = Math.min(sourceText.length, stop + 36);
  return (
    <blockquote className="evidence-snippet">
      {contextStart > 0 && <span aria-hidden="true">…</span>}
      <span>{sourceText.slice(contextStart, start)}</span>
      <mark>{sourceText.slice(start, stop)}</mark>
      <span>{sourceText.slice(stop, contextStop)}</span>
      {contextStop < sourceText.length && <span aria-hidden="true">…</span>}
      <small>
        Source characters {start.toLocaleString()}–{stop.toLocaleString()}
      </small>
    </blockquote>
  );
}

function EvidenceView({
  evidence,
  sourceText,
}: {
  evidence: ExtractionEvidence | null | undefined;
  sourceText: string;
}) {
  if (!evidence || evidence.fields.length === 0) {
    return (
      <div className="evidence-empty">
        <strong>No field evidence was returned</strong>
        <span>The extracted JSON is still available in the other tabs.</span>
      </div>
    );
  }
  const citations = new Map(evidence.citations.map((citation) => [citation.id, citation]));
  const hasCitationEvidence = evidence.citations.length > 0;
  return (
    <div className="evidence-view">
      <div className="evidence-meta" aria-label="AI Extract evidence metadata">
        <span>AI Extract {evidence.version || "2.1"}</span>
        <span>{evidence.mode === "precision" ? "Precision" : "Standard"}</span>
        <span>{evidence.fields.length.toLocaleString()} fields</span>
      </div>
      <div className="evidence-list">
        {evidence.fields.map((field, index) => {
          const fieldCitations = field.citation_ids
            .map((citationId) => citations.get(citationId))
            .filter((citation): citation is ExtractionCitation => Boolean(citation));
          return (
            <article className="evidence-field" key={`${field.path}-${index}`}>
              <div className="evidence-field-heading">
                <code>{field.path}</code>
                {typeof field.confidence_score === "number" && (
                  <span className="evidence-confidence">
                    {Math.round(field.confidence_score * 100)}% confidence
                  </span>
                )}
              </div>
              <div className="evidence-value">{formatEvidenceValue(field.value)}</div>
              {fieldCitations.length > 0 ? (
                fieldCitations.map((citation) => (
                  <EvidenceSnippet
                    key={citation.id}
                    citation={citation}
                    sourceText={sourceText}
                  />
                ))
              ) : hasCitationEvidence ? (
                <div className="evidence-no-citation">No source span returned.</div>
              ) : null}
            </article>
          );
        })}
      </div>
    </div>
  );
}

function CopyIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M5 2.5A1.5 1.5 0 0 1 6.5 1h5A1.5 1.5 0 0 1 13 2.5v7A1.5 1.5 0 0 1 11.5 11h-5A1.5 1.5 0 0 1 5 9.5z" />
      <path d="M3 5.5A1.5 1.5 0 0 1 4.5 4H5v1H4.5a.5.5 0 0 0-.5.5v7a.5.5 0 0 0 .5.5h5a.5.5 0 0 0 .5-.5V12h1v.5A1.5 1.5 0 0 1 9.5 14h-5A1.5 1.5 0 0 1 3 12.5z" />
    </svg>
  );
}

function JsonIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M6.25 2.5c-1.2 0-2 .67-2 1.9v1.2c0 .68-.17.94-.66 1.2l-.34.18v1.04l.34.18c.49.26.66.52.66 1.2v1.2c0 1.23.8 1.9 2 1.9h.75v-1h-.75c-.67 0-1-.3-1-.9v-1.2c0-.82-.23-1.34-.83-1.72.6-.38.83-.9.83-1.72V4.4c0-.6.33-.9 1-.9H7v-1zm3.5 0v1h.75c.67 0 1 .3 1 .9v1.2c0 .82.23 1.34.83 1.72-.6.38-.83.9-.83 1.72v1.2c0 .6-.33.9-1 .9H9v1h.75c1.2 0 2-.67 2-1.9v-1.2c0-.68.17-.94.66-1.2l.34-.18V6.78l-.34-.18c-.49-.26-.66-.52-.66-1.2V4.4c0-1.23-.8-1.9-2-1.9z" />
    </svg>
  );
}

function OptionalSettingsIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M4 2v12M12 2v12M2 5h4M10 11h4" />
      <circle cx="4" cy="5" r="1.75" />
      <circle cx="12" cy="11" r="1.75" />
    </svg>
  );
}

export default function ModelResultColumn({
  slot,
  models,
  sourceText,
  onModelChange,
  onRemove,
  onUpdateSlot,
  onRefreshModels,
  canRemove,
}: Props) {
  const { result, error, notice, activeTab, copied, loading } = slot;
  const [showOptionalSettings, setShowOptionalSettings] = useState(false);
  const optionalSettingsRef = useRef<HTMLDivElement>(null);
  const optionalSettingsButtonRef = useRef<HTMLButtonElement>(null);
  const customModels = models.filter((model) => model.model_type === "custom");
  const fmapiModels = models.filter((model) => model.model_type === "fmapi");
  const aiExtractModels = models.filter((model) => model.model_type === "ai_extract");
  const selectedModelInfo = models.find((model) => model.name === slot.selectedModel);
  const selectedDisplayName = displayModelName(selectedModelInfo, slot.selectedModel);
  const selectedCustomModel = selectedModelInfo?.model_type === "custom";
  const selectedAiExtract = selectedModelInfo?.model_type === "ai_extract";
  const selectedServingModel = Boolean(selectedModelInfo && !selectedAiExtract);
  const enabledAiExtractOptionCount = [
    slot.aiExtractPrecision,
    slot.aiExtractCitations,
    slot.aiExtractConfidenceScores,
  ].filter(Boolean).length;
  const outputTokenOverrideValidation = validateOutputTokenOverride(
    slot.maxOutputTokensOverride
  );
  const hasOutputTokenOverride =
    selectedServingModel && slot.maxOutputTokensOverride.trim().length > 0;
  const hasOptionalSettingsConfigured = selectedAiExtract
    ? enabledAiExtractOptionCount > 0
    : hasOutputTokenOverride;
  const optionalSettingsLabel = selectedAiExtract
    ? `AI Extract optional settings${
        enabledAiExtractOptionCount > 0
          ? `, ${enabledAiExtractOptionCount} enabled`
          : ""
      }`
    : `Optional settings for ${selectedDisplayName}${
        outputTokenOverrideValidation.error
          ? ", output limit is invalid"
          : hasOutputTokenOverride
            ? ", output limit override set"
            : ""
      }`;
  const selectedServingRuntime = selectedModelInfo?.serving_runtime ?? null;
  const selectedModelUnavailable = Boolean(
    selectedModelInfo && selectedModelInfo.availability_status !== "ready"
  );
  const unavailableMessage =
    selectedModelInfo?.availability_message ||
    "This model is not currently verified as ready. Refresh its status or choose another model.";
  const hasOutputWarning = Boolean(result && (result.truncated || result.repaired_output));
  const finishReasonLabel =
    result?.finish_reason && result.finish_reason !== "stop"
      ? ` (finish_reason=${result.finish_reason})`
      : "";
  const servingModePillLabel =
    selectedServingRuntime === "vllm_beta"
      ? "vLLM Beta"
      : selectedServingRuntime === "standard"
        ? "Standard serving"
        : null;
  const servingModeLabel =
    selectedServingRuntime === "vllm_beta"
      ? "vLLM-based serving in Beta"
      : selectedServingRuntime === "standard"
        ? "vLLM not used for this model"
        : null;
  const showModelStatusRow =
    selectedAiExtract ||
    (selectedCustomModel && Boolean(servingModePillLabel && servingModeLabel)) ||
    selectedModelUnavailable;

  const handleCopy = () => {
    if (!result) return;
    const text =
      typeof result.prediction === "string"
        ? result.prediction
        : JSON.stringify(result.prediction, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      onUpdateSlot({ copied: true });
      setTimeout(() => onUpdateSlot({ copied: false }), 1500);
    });
  };

  const handleDownloadJSON = () => {
    if (!result) return;
    const text =
      typeof result.prediction === "string"
        ? result.prediction
        : JSON.stringify(result.prediction, null, 2);
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${slot.selectedModel}-output.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formattedTabId = `${slot.id}-formatted-tab`;
  const rawTabId = `${slot.id}-raw-tab`;
  const evidenceTabId = `${slot.id}-evidence-tab`;
  const resultPanelId = `${slot.id}-result-panel`;
  const hasConfidenceEvidence = Boolean(
    result?.evidence?.fields.some(
      (field) => typeof field.confidence_score === "number"
    )
  );
  const hasCitationEvidence = Boolean(
    result?.evidence?.fields.some((field) => field.citation_ids.length > 0)
  );
  const showEvidenceTab = Boolean(
    result?.runtime === "ai_extract" &&
      (hasConfidenceEvidence || hasCitationEvidence)
  );
  const activeTabId =
    activeTab === "formatted"
      ? formattedTabId
      : activeTab === "evidence"
        ? evidenceTabId
        : rawTabId;

  useEffect(() => {
    setShowOptionalSettings(false);
  }, [slot.selectedModel]);

  useEffect(() => {
    if (!showOptionalSettings) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (
        optionalSettingsRef.current &&
        !optionalSettingsRef.current.contains(event.target as Node)
      ) {
        setShowOptionalSettings(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setShowOptionalSettings(false);
      optionalSettingsButtonRef.current?.focus();
    };

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [showOptionalSettings]);

  return (
    <section
      className="model-column"
      aria-label={`Output for ${selectedDisplayName}`}
      aria-busy={loading}
    >
      <div className="model-column-header">
        <select
          id={`${slot.id}-model`}
          name={`${slot.id}-model`}
          value={slot.selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          onFocus={onRefreshModels}
          aria-label="Select a model"
          disabled={loading}
        >
          {aiExtractModels.length > 0 && (
            <optgroup label="Agent Bricks · Information Extraction">
              {aiExtractModels.map((model) => (
                <option key={model.name} value={model.name}>
                  AI Extract
                </option>
              ))}
            </optgroup>
          )}
          {fmapiModels.length > 0 && (
            <optgroup label="Databricks FMAPI">
              {fmapiModels.map((model) => (
                <option key={model.name} value={model.name}>
                  {displayModelName(model, model.name)}
                  {model.availability_status !== "ready" ? " (unavailable)" : ""}
                </option>
              ))}
            </optgroup>
          )}
          {customModels.length > 0 && (
            <optgroup label="Custom models">
              {customModels.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name}
                  {model.availability_status !== "ready" ? " (unavailable)" : ""}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        {selectedModelInfo && (
          <div className="model-optional-settings" ref={optionalSettingsRef}>
            <button
              ref={optionalSettingsButtonRef}
              type="button"
              className={`model-options-button ${showOptionalSettings ? "is-open" : ""} ${
                hasOptionalSettingsConfigured ? "is-configured" : ""
              } ${
                outputTokenOverrideValidation.error ? "has-error" : ""
              }`}
              aria-label={optionalSettingsLabel}
              aria-expanded={showOptionalSettings}
              aria-controls={`${slot.id}-optional-settings`}
              title="Optional settings"
              onClick={() => setShowOptionalSettings((current) => !current)}
              disabled={loading || selectedModelUnavailable}
            >
              <OptionalSettingsIcon />
              {hasOptionalSettingsConfigured && (
                <span className="model-options-indicator" aria-hidden="true" />
              )}
            </button>
            {showOptionalSettings && (
              <div
                id={`${slot.id}-optional-settings`}
                className="model-options-popover"
                role="dialog"
                aria-label={`Optional settings for ${selectedDisplayName}`}
              >
                {selectedAiExtract ? (
                  <fieldset
                    className="ai-extract-options"
                    disabled={loading || selectedModelUnavailable}
                  >
                    <legend className="sr-only">AI Extract optional settings</legend>
                    <label className="ai-extract-option-row">
                      <span className="ai-extract-option-copy">
                        <span
                          className="ai-extract-option-title"
                          id={`${slot.id}-precision-label`}
                        >
                          Precision mode
                        </span>
                        <span
                          className="ai-extract-option-description"
                          id={`${slot.id}-precision-description`}
                        >
                          Higher accuracy for nuanced fields and long documents
                        </span>
                      </span>
                      <input
                        className="ai-extract-option-switch"
                        type="checkbox"
                        role="switch"
                        checked={slot.aiExtractPrecision}
                        onChange={(event) =>
                          onUpdateSlot({ aiExtractPrecision: event.target.checked })
                        }
                        aria-labelledby={`${slot.id}-precision-label`}
                        aria-describedby={`${slot.id}-precision-description`}
                      />
                    </label>
                    <label className="ai-extract-option-row">
                      <span className="ai-extract-option-copy">
                        <span
                          className="ai-extract-option-title"
                          id={`${slot.id}-citations-label`}
                        >
                          Citations
                        </span>
                        <span
                          className="ai-extract-option-description"
                          id={`${slot.id}-citations-description`}
                        >
                          Show source references for extracted values
                        </span>
                      </span>
                      <input
                        className="ai-extract-option-switch"
                        type="checkbox"
                        role="switch"
                        checked={slot.aiExtractCitations}
                        onChange={(event) =>
                          onUpdateSlot({ aiExtractCitations: event.target.checked })
                        }
                        aria-labelledby={`${slot.id}-citations-label`}
                        aria-describedby={`${slot.id}-citations-description`}
                      />
                    </label>
                    <label className="ai-extract-option-row">
                      <span className="ai-extract-option-copy">
                        <span
                          className="ai-extract-option-title"
                          id={`${slot.id}-confidence-label`}
                        >
                          Confidence scores
                        </span>
                        <span
                          className="ai-extract-option-description"
                          id={`${slot.id}-confidence-description`}
                        >
                          Show confidence levels for extracted values
                        </span>
                      </span>
                      <input
                        className="ai-extract-option-switch"
                        type="checkbox"
                        role="switch"
                        checked={slot.aiExtractConfidenceScores}
                        onChange={(event) =>
                          onUpdateSlot({
                            aiExtractConfidenceScores: event.target.checked,
                          })
                        }
                        aria-labelledby={`${slot.id}-confidence-label`}
                        aria-describedby={`${slot.id}-confidence-description`}
                      />
                    </label>
                  </fieldset>
                ) : (
                  <fieldset
                    className="serving-model-options"
                    disabled={loading || selectedModelUnavailable}
                  >
                    <legend className="sr-only">Serving model optional settings</legend>
                    <label
                      className="serving-model-option-label"
                      htmlFor={`${slot.id}-max-output-tokens`}
                    >
                      Output limit
                    </label>
                    <input
                      id={`${slot.id}-max-output-tokens`}
                      name={`${slot.id}-max-output-tokens`}
                      className="serving-model-option-input"
                      type="number"
                      inputMode="numeric"
                      min={1}
                      max={MAX_OUTPUT_TOKENS_CEILING}
                      step={1}
                      value={slot.maxOutputTokensOverride}
                      placeholder="Model default"
                      onChange={(event) =>
                        onUpdateSlot({ maxOutputTokensOverride: event.target.value })
                      }
                      aria-invalid={Boolean(outputTokenOverrideValidation.error)}
                      aria-describedby={`${slot.id}-max-output-tokens-hint${
                        outputTokenOverrideValidation.error
                          ? ` ${slot.id}-max-output-tokens-error`
                          : ""
                      }`}
                      aria-errormessage={
                        outputTokenOverrideValidation.error
                          ? `${slot.id}-max-output-tokens-error`
                          : undefined
                      }
                    />
                    <span
                      className="serving-model-option-hint"
                      id={`${slot.id}-max-output-tokens-hint`}
                    >
                      Blank uses this model&apos;s configured default
                      {selectedModelInfo.max_tokens != null
                        ? ` (${selectedModelInfo.max_tokens.toLocaleString()})`
                        : ""}
                      . Allowed range: 1–{MAX_OUTPUT_TOKENS_CEILING.toLocaleString()} tokens.
                    </span>
                    {outputTokenOverrideValidation.error && (
                      <span
                        className="serving-model-option-error"
                        id={`${slot.id}-max-output-tokens-error`}
                      >
                        {outputTokenOverrideValidation.error}
                      </span>
                    )}
                  </fieldset>
                )}
              </div>
            )}
          </div>
        )}
        {canRemove && (
          <button
            className="btn-remove-slot"
            onClick={onRemove}
            aria-label="Remove model column"
          >
            &times;
          </button>
        )}
      </div>
      {selectedServingModel &&
        outputTokenOverrideValidation.error &&
        !showOptionalSettings && (
          <div className="model-settings-validation" role="alert">
            Output limit: {outputTokenOverrideValidation.error}
          </div>
        )}
      {showModelStatusRow && (
        <div className="model-status-row">
          <div className="model-status-helper">
            {selectedAiExtract && (
              <div className="model-status-summary">
                <span className="model-status-pill is-ai-extract">AI Extract 2.1</span>
                <span className="model-status-label">
                  Managed extraction with optional precision and evidence
                </span>
              </div>
            )}
            {servingModePillLabel && servingModeLabel && (
              <div className="model-status-summary">
                <span
                  className={`model-status-pill ${
                    selectedServingRuntime === "vllm_beta"
                      ? "is-vllm"
                      : "is-standard"
                  }`}
                >
                  {servingModePillLabel}
                </span>
                <span className="model-status-label">{servingModeLabel}</span>
              </div>
            )}
            {selectedModelUnavailable && (
              <div className="model-status-message">
                {unavailableMessage}
              </div>
            )}
          </div>
          {selectedModelUnavailable && (
            <button
              type="button"
              className="btn-status-refresh"
              onClick={onRefreshModels}
            >
              Retry status
            </button>
          )}
        </div>
      )}

      {loading && (
        <div className="model-column-loading" role="status" aria-live="polite">
          <span className="spinner spinner-lg" aria-hidden="true" />
          <div className="model-loading-copy">
            <strong>Extracting structured data</strong>
            <span>{selectedDisplayName}</span>
          </div>
          <div className="result-skeleton" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
        </div>
      )}

      {!loading && error && (
        <div className="error" role="alert">
          <div className="error-title">{error.message}</div>
          {error.detail && <div className="error-detail">{error.detail}</div>}
        </div>
      )}

      {!loading && !result && !error && notice && (
        <div className="result-warning result-warning-neutral" role="status">
          <div>{notice}</div>
        </div>
      )}

      {!loading && result && (
        <>
          <div className="sr-only" role="status">
            Extraction complete for {selectedDisplayName}.
          </div>
          <div className="result-header">
            <div className="tabs" role="tablist">
              <button
                id={formattedTabId}
                className={`tab ${activeTab === "formatted" ? "active" : ""}`}
                role="tab"
                aria-selected={activeTab === "formatted"}
                aria-controls={resultPanelId}
                onClick={() => onUpdateSlot({ activeTab: "formatted" })}
              >
                Formatted
              </button>
              <button
                id={rawTabId}
                className={`tab ${activeTab === "raw" ? "active" : ""}`}
                role="tab"
                aria-selected={activeTab === "raw"}
                aria-controls={resultPanelId}
                onClick={() => onUpdateSlot({ activeTab: "raw" })}
              >
                JSON
              </button>
              {showEvidenceTab && (
                <button
                  id={evidenceTabId}
                  className={`tab ${activeTab === "evidence" ? "active" : ""}`}
                  role="tab"
                  aria-selected={activeTab === "evidence"}
                  aria-controls={resultPanelId}
                  onClick={() => onUpdateSlot({ activeTab: "evidence" })}
                >
                  Evidence
                </button>
              )}
            </div>
            <div className="result-actions">
              <button className="btn-copy" onClick={handleCopy} aria-label="Copy result">
                <span className="btn-action-icon"><CopyIcon /></span>
                <span>{copied ? "Copied!" : "Copy"}</span>
              </button>
              <button className="btn-export" onClick={handleDownloadJSON} aria-label="Download JSON result">
                <span className="btn-action-icon"><JsonIcon /></span>
                <span>JSON</span>
              </button>
            </div>
          </div>

          <div
            id={resultPanelId}
            className="result-box"
            role="tabpanel"
            aria-labelledby={activeTabId}
          >
            {activeTab === "evidence" ? (
              <EvidenceView evidence={result.evidence} sourceText={sourceText} />
            ) : activeTab === "raw" ? (
              <pre>{result.raw_output}</pre>
            ) : (
              <div className="formatted-output">
                <FormattedView data={result.prediction} />
              </div>
            )}
          </div>

          {hasOutputWarning && (
            <div className="result-warning" role="status">
              {result.truncated && (
                <div>
                  Output hit the token ceiling{finishReasonLabel}. Treat this result as incomplete.
                </div>
              )}
              {result.repaired_output && (
                <div>
                  The app repaired malformed JSON before rendering it. Use the raw output tab for inspection.
                </div>
              )}
            </div>
          )}
          {result.schema_warnings && result.schema_warnings.length > 0 && (
            <div className="result-warning result-warning-neutral" role="status">
              {result.schema_warnings.map((warning) => (
                <div key={warning}>{warning}</div>
              ))}
            </div>
          )}

          <details className="result-details">
            <summary>Details</summary>
            <div className="latency" aria-label="Inference details">
              <div className="result-meta">
                {result.model && (
                  <span className="result-meta-model">{selectedDisplayName}</span>
                )}
                <span><strong>{result.latency_ms.toFixed(0)}</strong> ms latency</span>
                {typeof result.output_tokens_per_second === "number" && (
                  <span><strong>{result.output_tokens_per_second.toFixed(1)}</strong> tok/s*</span>
                )}
                {slot.outputTokenLimit != null && (
                  <span>
                    <strong>{slot.outputTokenLimit.toLocaleString()}</strong>{" "}
                    {slot.outputTokenLimitSource === "override"
                      ? "request output cap"
                      : "configured output cap"}
                  </span>
                )}
                {typeof result.prompt_tokens === "number" && (
                  <span><strong>{result.prompt_tokens}</strong> prompt</span>
                )}
                {typeof result.completion_tokens === "number" && (
                  <span><strong>{result.completion_tokens}</strong> completion</span>
                )}
                {typeof result.total_tokens === "number" && (
                  <span><strong>{result.total_tokens}</strong> total</span>
                )}
              </div>
              {typeof result.output_tokens_per_second === "number" && (
                <div className="latency-note">
                  * Based on end-to-end latency; includes network and serving overhead.
                </div>
              )}
              {slot.outputTokenLimit != null && (
                <div className="latency-note">
                  {slot.outputTokenLimitSource === "override" ? (
                    <>
                      The app sends this request-specific cap as <code>max_tokens</code>. The
                      endpoint may enforce a lower provider or model ceiling.
                    </>
                  ) : (
                    <>
                      This is the model&apos;s configured default output cap. The endpoint may
                      enforce a lower provider or model ceiling.
                    </>
                  )}
                </div>
              )}
            </div>
          </details>
        </>
      )}

      {!loading && !result && !error && !notice && (
        <div className="placeholder">
          <div className="placeholder-icon" aria-hidden="true">{"{}"}</div>
          <strong>Ready for structured output</strong>
          <span>Choose a preset or define a schema, add medical text, then run extraction.</span>
        </div>
      )}
    </section>
  );
}
