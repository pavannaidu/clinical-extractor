import { useCallback, useEffect, useMemo, useState } from "react";
import {
  cancelEvaluationRun,
  createEvaluationRun,
  fetchEvaluationCatalogs,
  fetchEvaluationColumns,
  fetchEvaluationReport,
  fetchEvaluationRuns,
  fetchEvaluationSchemas,
  fetchEvaluationTables,
  preflightEvaluation,
  rerunEvaluation,
} from "./api";
import type {
  EvaluationColumn,
  EvaluationCreateRequest,
  EvaluationPreflightResponse,
  EvaluationReport,
  EvaluationRun,
  ModelInfo,
  Preset,
} from "./types";

interface EvaluationsViewProps {
  models: ModelInfo[];
  presets: Preset[];
}

interface EvaluationDraft {
  name: string;
  catalog: string;
  schema: string;
  tableName: string;
  inputColumn: string;
  groundTruthColumn: string;
  identifierColumn: string;
  model: string;
  presetId: string;
  rowLimit: string;
  enableGroundedness: boolean;
  enableCompleteness: boolean;
}

const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "interrupted",
]);

function getErrorMessage(error: unknown, fallback: string): string {
  if (error && typeof error === "object") {
    if ("message" in error && typeof error.message === "string") {
      const detail =
        "detail" in error && typeof error.detail === "string" ? error.detail : "";
      return detail ? `${error.message} ${detail}` : error.message;
    }
  }
  return error instanceof Error ? error.message : fallback;
}

function formatPercent(value: number | null): string {
  return value == null || !Number.isFinite(value)
    ? "—"
    : `${Math.round(value * 100)}%`;
}

function formatLatency(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`;
}

function formatDate(value: string): string {
  return new Date(value).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function runIsActive(run: EvaluationRun): boolean {
  return !TERMINAL_STATUSES.has(run.status);
}

function makeDefaultDraft(models: ModelInfo[], presets: Preset[]): EvaluationDraft {
  return {
    name: `Clinical extraction evaluation · ${new Date().toLocaleDateString()}`,
    catalog: "",
    schema: "",
    tableName: "",
    inputColumn: "",
    groundTruthColumn: "",
    identifierColumn: "",
    model:
      models.find(
        (model) =>
          model.model_type !== "ai_extract" && model.availability_status === "ready"
      )?.name ?? "",
    presetId: presets[0]?.id ?? "",
    rowLimit: "100",
    enableGroundedness: false,
    enableCompleteness: false,
  };
}

function RunStatus({ run }: { run: EvaluationRun }) {
  return (
    <span className={`evaluation-status is-${run.status}`}>
      <i aria-hidden="true" />
      {run.status}
    </span>
  );
}

function KpiCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <article className="evaluation-kpi-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

export default function EvaluationsView({ models, presets }: EvaluationsViewProps) {
  const evaluationModels = useMemo(
    () => models.filter((model) => model.model_type !== "ai_extract"),
    [models]
  );
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [reportLoading, setReportLoading] = useState(false);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [setupError, setSetupError] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [draft, setDraft] = useState<EvaluationDraft>(() =>
    makeDefaultDraft(evaluationModels, presets)
  );
  const [catalogs, setCatalogs] = useState<string[]>([]);
  const [schemas, setSchemas] = useState<string[]>([]);
  const [tables, setTables] = useState<string[]>([]);
  const [columns, setColumns] = useState<EvaluationColumn[]>([]);
  const [discoveryLoading, setDiscoveryLoading] = useState(false);
  const [preflight, setPreflight] = useState<EvaluationPreflightResponse | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [rerunning, setRerunning] = useState(false);

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) ?? report?.run ?? null,
    [report, runs, selectedRunId]
  );
  const selectedPreset = presets.find((preset) => preset.id === draft.presetId);
  const activeRun = runs.find(runIsActive) ?? null;
  const canCreate = !activeRun;

  const loadRuns = useCallback(async (preserveSelection = true) => {
    try {
      const nextRuns = await fetchEvaluationRuns();
      setRuns(nextRuns);
      setSelectedRunId((current) => {
        if (preserveSelection && nextRuns.some((run) => run.run_id === current)) {
          return current;
        }
        return nextRuns[0]?.run_id ?? "";
      });
      setError("");
      return nextRuns;
    } catch (nextError) {
      setError(getErrorMessage(nextError, "Unable to load evaluation history."));
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const loadReport = useCallback(async (runId: string, quiet = false) => {
    if (!runId) {
      setReport(null);
      return;
    }
    if (!quiet) setReportLoading(true);
    try {
      const nextReport = await fetchEvaluationReport(runId);
      setReport(nextReport);
      setRuns((current) =>
        current.map((run) =>
          run.run_id === nextReport.run.run_id ? nextReport.run : run
        )
      );
      setActionError("");
    } catch (nextError) {
      if (!quiet) {
        setActionError(getErrorMessage(nextError, "Unable to load this report."));
      }
    } finally {
      if (!quiet) setReportLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRuns(false).catch(() => undefined);
  }, [loadRuns]);

  useEffect(() => {
    if (selectedRunId) loadReport(selectedRunId).catch(() => undefined);
    else setReport(null);
  }, [loadReport, selectedRunId]);

  useEffect(() => {
    if (!activeRun) return;
    const timer = window.setInterval(async () => {
      const nextRuns = await loadRuns();
      const nextActive = nextRuns.find((run) => run.run_id === activeRun.run_id);
      if (selectedRunId === activeRun.run_id) {
        await loadReport(activeRun.run_id, true);
      }
      if (nextActive && TERMINAL_STATUSES.has(nextActive.status)) {
        window.clearInterval(timer);
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [activeRun?.run_id, loadReport, loadRuns, selectedRunId]);

  useEffect(() => {
    setDraft((current) => {
      const next = { ...current };
      if (!next.model && evaluationModels.length) {
        next.model =
          evaluationModels.find((model) => model.availability_status === "ready")
            ?.name ?? evaluationModels[0].name;
      }
      if (!next.presetId && presets.length) next.presetId = presets[0].id;
      return next;
    });
  }, [evaluationModels, presets]);

  const openCreate = async () => {
    setShowCreate(true);
    setSetupError("");
    setPreflight(null);
    if (catalogs.length) return;
    setDiscoveryLoading(true);
    try {
      setCatalogs(await fetchEvaluationCatalogs());
    } catch (nextError) {
      setSetupError(getErrorMessage(nextError, "Unable to discover Unity Catalog."));
    } finally {
      setDiscoveryLoading(false);
    }
  };

  const closeCreate = () => {
    setShowCreate(false);
    setSetupError("");
  };

  const updateDraft = <Key extends keyof EvaluationDraft>(
    key: Key,
    value: EvaluationDraft[Key]
  ) => {
    setDraft((current) => ({ ...current, [key]: value }));
    setPreflight(null);
    setSetupError("");
  };

  const handleCatalogChange = async (catalog: string) => {
    setDraft((current) => ({
      ...current,
      catalog,
      schema: "",
      tableName: "",
      inputColumn: "",
      groundTruthColumn: "",
      identifierColumn: "",
    }));
    setSchemas([]);
    setTables([]);
    setColumns([]);
    setPreflight(null);
    if (!catalog) return;
    setDiscoveryLoading(true);
    try {
      setSchemas(await fetchEvaluationSchemas(catalog));
    } catch (nextError) {
      setSetupError(getErrorMessage(nextError, "Unable to list schemas."));
    } finally {
      setDiscoveryLoading(false);
    }
  };

  const handleSchemaChange = async (schema: string) => {
    setDraft((current) => ({
      ...current,
      schema,
      tableName: "",
      inputColumn: "",
      groundTruthColumn: "",
      identifierColumn: "",
    }));
    setTables([]);
    setColumns([]);
    setPreflight(null);
    if (!schema || !draft.catalog) return;
    setDiscoveryLoading(true);
    try {
      setTables(await fetchEvaluationTables(draft.catalog, schema));
    } catch (nextError) {
      setSetupError(getErrorMessage(nextError, "Unable to list tables."));
    } finally {
      setDiscoveryLoading(false);
    }
  };

  const handleTableChange = async (tableName: string) => {
    setDraft((current) => ({
      ...current,
      tableName,
      inputColumn: "",
      groundTruthColumn: "",
      identifierColumn: "",
    }));
    setColumns([]);
    setPreflight(null);
    if (!tableName) return;
    setDiscoveryLoading(true);
    try {
      setColumns(await fetchEvaluationColumns(tableName));
    } catch (nextError) {
      setSetupError(getErrorMessage(nextError, "Unable to describe the table."));
    } finally {
      setDiscoveryLoading(false);
    }
  };

  const datasetRequest = useMemo(() => {
    if (
      !selectedPreset ||
      !draft.tableName ||
      !draft.inputColumn ||
      !draft.groundTruthColumn
    ) {
      return null;
    }
    return {
      table_name: draft.tableName,
      input_column: draft.inputColumn,
      ground_truth_column: draft.groundTruthColumn,
      identifier_column: draft.identifierColumn || null,
      schema_json: selectedPreset.schema_json,
      row_limit: Math.min(100, Math.max(1, Number.parseInt(draft.rowLimit, 10) || 100)),
    };
  }, [draft, selectedPreset]);

  const handlePreflight = async () => {
    if (!datasetRequest) return;
    setPreflightLoading(true);
    setSetupError("");
    try {
      setPreflight(await preflightEvaluation(datasetRequest));
    } catch (nextError) {
      setPreflight(null);
      setSetupError(getErrorMessage(nextError, "Dataset validation failed."));
    } finally {
      setPreflightLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!datasetRequest || !selectedPreset || !preflight || preflight.invalid_rows > 0) {
      return;
    }
    const request: EvaluationCreateRequest = {
      ...datasetRequest,
      name: draft.name.trim(),
      model: draft.model,
      preset_id: selectedPreset.id,
      system_prompt: selectedPreset.system_prompt,
      enable_groundedness: draft.enableGroundedness,
      enable_completeness: draft.enableCompleteness,
    };
    setCreating(true);
    setSetupError("");
    try {
      const created = await createEvaluationRun(request);
      setRuns((current) => [created, ...current]);
      setSelectedRunId(created.run_id);
      setShowCreate(false);
      setPreflight(null);
    } catch (nextError) {
      setSetupError(getErrorMessage(nextError, "Unable to start the evaluation."));
    } finally {
      setCreating(false);
    }
  };

  const handleCancel = async () => {
    if (!selectedRun || !runIsActive(selectedRun)) return;
    setCancelling(true);
    setActionError("");
    try {
      const updated = await cancelEvaluationRun(selectedRun.run_id);
      setRuns((current) =>
        current.map((run) => (run.run_id === updated.run_id ? updated : run))
      );
      await loadReport(updated.run_id, true);
    } catch (nextError) {
      setActionError(getErrorMessage(nextError, "Unable to cancel the run."));
    } finally {
      setCancelling(false);
    }
  };

  const handleRerun = async () => {
    if (!selectedRun || activeRun) return;
    setRerunning(true);
    setActionError("");
    try {
      const created = await rerunEvaluation(selectedRun.run_id);
      setRuns((current) => [created, ...current]);
      setSelectedRunId(created.run_id);
    } catch (nextError) {
      setActionError(getErrorMessage(nextError, "Unable to rerun this evaluation."));
    } finally {
      setRerunning(false);
    }
  };

  const progress = selectedRun?.total_rows
    ? Math.round((selectedRun.completed_rows / selectedRun.total_rows) * 100)
    : 0;
  const fieldScores = Object.entries(report?.run.metrics.field_scores ?? {}).sort(
    ([leftPath, leftScore], [rightPath, rightScore]) =>
      leftScore - rightScore || leftPath.localeCompare(rightPath)
  );
  const discoveryBlocked = Boolean(
    setupError && !discoveryLoading && catalogs.length === 0
  );

  return (
    <main id="main-workspace" className="evaluations-view">
      <section className="evaluations-header">
        <div>
          <h2>Evaluations <span>[Preview]</span></h2>
          <p>Compare model output with ground truth.</p>
        </div>
        <button
          type="button"
          className="btn-submit evaluation-new-button"
          onClick={() => openCreate().catch(() => undefined)}
          disabled={!canCreate || evaluationModels.length === 0 || presets.length === 0}
          title={activeRun ? "Finish or cancel the active run first" : undefined}
        >
          + New evaluation
        </button>
      </section>

      {error && (
        <section className="evaluation-alert is-error" role="alert">
          <strong>Evaluation history is unavailable</strong>
          <span>{error}</span>
          <button type="button" onClick={() => loadRuns(false)}>Retry</button>
        </section>
      )}
      {actionError && !showCreate && (
        <section className="evaluation-alert is-error" role="alert">
          <strong>Action could not be completed</strong>
          <span>{actionError}</span>
        </section>
      )}

      <div className="evaluations-layout">
        <aside className="evaluation-history" aria-label="Evaluation history">
          <div className="evaluation-section-heading">
            <h3>Runs</h3>
            <span className="evaluation-count">{runs.length}</span>
          </div>
          {loading ? (
            <div className="evaluation-history-state" role="status">Loading runs…</div>
          ) : runs.length === 0 ? (
            <div className="evaluation-history-state">
              <strong>No evaluations yet</strong>
              <p>Validate a governed dataset and create the first report card.</p>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => openCreate().catch(() => undefined)}
                disabled={!canCreate}
              >
                Create evaluation
              </button>
            </div>
          ) : (
            <div className="evaluation-run-list">
              {runs.map((run) => (
                <button
                  key={run.run_id}
                  type="button"
                  className={`evaluation-run-item ${selectedRunId === run.run_id ? "active" : ""}`}
                  onClick={() => setSelectedRunId(run.run_id)}
                >
                  <span className="evaluation-run-item-topline">
                    <strong>{run.name}</strong>
                    <RunStatus run={run} />
                  </span>
                  <span>{run.model}</span>
                  <small>{formatDate(run.created_at)} · {run.completed_rows}/{run.total_rows || run.row_limit} rows</small>
                </button>
              ))}
            </div>
          )}
        </aside>

        <section className="evaluation-report" aria-live="polite">
          {reportLoading && !report ? (
            <div className="evaluation-empty-report" role="status">Loading report…</div>
          ) : !selectedRun ? (
            <div className="evaluation-empty-report">
              <span className="evaluation-empty-icon" aria-hidden="true">✓</span>
              <h3>Quality evidence will appear here</h3>
              <p>Choose a completed run or create an evaluation from a Unity Catalog table.</p>
            </div>
          ) : (
            <>
              <div className="evaluation-report-header">
                <div>
                  <div className="evaluation-report-titleline">
                    <RunStatus run={selectedRun} />
                    <span>{formatDate(selectedRun.created_at)}</span>
                  </div>
                  <h3>{selectedRun.name}</h3>
                  <p>{selectedRun.table_name} · {selectedRun.model}</p>
                </div>
                <div className="evaluation-report-actions">
                  {report?.mlflow_url && (
                    <a
                      className="btn-secondary evaluation-mlflow-link"
                      href={report.mlflow_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open in MLflow ↗
                    </a>
                  )}
                  {runIsActive(selectedRun) ? (
                    <button
                      type="button"
                      className="btn-secondary evaluation-danger-button"
                      onClick={handleCancel}
                      disabled={cancelling}
                    >
                      {cancelling ? "Cancelling…" : "Cancel run"}
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={handleRerun}
                      disabled={Boolean(activeRun) || rerunning}
                    >
                      {rerunning ? "Starting…" : "Rerun"}
                    </button>
                  )}
                </div>
              </div>

              {runIsActive(selectedRun) && (
                <section className="evaluation-progress-card">
                  <div>
                    <strong>{selectedRun.status === "queued" ? "Queued" : "Running extraction"}</strong>
                    <span>{selectedRun.completed_rows} of {selectedRun.total_rows || selectedRun.row_limit} records complete</span>
                  </div>
                  <div
                    className="evaluation-progress-track"
                    role="progressbar"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={progress}
                  >
                    <span style={{ width: `${progress}%` }} />
                  </div>
                  <small>{progress}% · {selectedRun.succeeded_rows} succeeded · {selectedRun.failed_rows} failed</small>
                </section>
              )}

              {(selectedRun.status === "failed" || selectedRun.status === "interrupted") && (
                <section className="evaluation-alert is-error" role="alert">
                  <strong>{selectedRun.status === "interrupted" ? "Run interrupted" : "Evaluation failed"}</strong>
                  <span>{selectedRun.error_message || "No additional error detail is available."}</span>
                </section>
              )}

              {selectedRun.status === "cancelled" && (
                <section className="evaluation-alert">
                  <strong>Run cancelled</strong>
                  <span>Completed document summaries remain available below.</span>
                </section>
              )}

              <section className="evaluation-kpi-grid" aria-label="Evaluation metrics">
                <KpiCard
                  label="Leaf-field accuracy"
                  value={formatPercent(selectedRun.metrics.overall_field_accuracy)}
                  detail="Exact field matches"
                />
                <KpiCard
                  label="Fully correct documents"
                  value={formatPercent(selectedRun.metrics.fully_correct_percentage)}
                  detail={`${selectedRun.completed_rows} evaluated`}
                />
                <KpiCard
                  label="Schema-valid outputs"
                  value={formatPercent(selectedRun.metrics.schema_valid_percentage)}
                  detail="Valid preset schema"
                />
                <KpiCard
                  label="Inference success"
                  value={formatPercent(selectedRun.metrics.inference_success_rate)}
                  detail={`Avg latency ${formatLatency(selectedRun.metrics.average_latency_ms)}`}
                />
              </section>

              {Object.keys(selectedRun.metrics.judge_metrics).length > 0 && (
                <section className="evaluation-card">
                  <div className="evaluation-card-heading">
                    <div>
                      <span className="evaluation-eyebrow">Optional MLflow judges</span>
                      <h4>Semantic quality</h4>
                    </div>
                  </div>
                  <div className="evaluation-judge-grid">
                    {Object.entries(selectedRun.metrics.judge_metrics).map(([name, value]) => (
                      <div key={name}>
                        <span>{name.replace(/_/g, " ")}</span>
                        <strong>{typeof value === "number" ? formatPercent(value) : String(value)}</strong>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              <section className="evaluation-card">
                <div className="evaluation-card-heading">
                  <div>
                    <h4>Field accuracy</h4>
                    <span className="evaluation-section-note">Weakest fields first</span>
                  </div>
                  <span className="evaluation-count">{fieldScores.length}</span>
                </div>
                {fieldScores.length === 0 ? (
                  <p className="evaluation-card-empty">Field scores appear after the run completes.</p>
                ) : (
                  <div className="evaluation-field-score-list">
                    {fieldScores.map(([path, score]) => (
                      <div className="evaluation-field-score" key={path}>
                        <code>{path}</code>
                        <div className="evaluation-field-score-track" aria-hidden="true">
                          <span style={{ width: `${Math.max(0, Math.min(100, score * 100))}%` }} />
                        </div>
                        <strong>{formatPercent(score)}</strong>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section className="evaluation-card">
                <div className="evaluation-card-heading">
                  <div>
                    <h4>Documents</h4>
                    <span className="evaluation-section-note">Mismatch details</span>
                  </div>
                  <span className="evaluation-count">{report?.results.length ?? 0}</span>
                </div>
                {!report?.results.length ? (
                  <p className="evaluation-card-empty">Document summaries appear as records finish.</p>
                ) : (
                  <div className="evaluation-table-scroll" tabIndex={0}>
                    <table className="evaluation-table">
                      <thead>
                        <tr>
                          <th scope="col">Record</th>
                          <th scope="col">Field match</th>
                          <th scope="col">Schema</th>
                          <th scope="col">Latency</th>
                          <th scope="col">Mismatches</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.results.map((result) => (
                          <tr key={result.record_id}>
                            <td><code>{result.record_id}</code></td>
                            <td>{result.matched_fields}/{result.total_fields}</td>
                            <td>
                              <span className={`evaluation-table-state ${result.schema_valid ? "is-valid" : "is-invalid"}`}>
                                {result.schema_valid ? "Valid" : "Invalid"}
                              </span>
                            </td>
                            <td>{formatLatency(result.latency_ms)}</td>
                            <td>
                              {result.error_message ? (
                                <span className="evaluation-error-text">{result.error_message}</span>
                              ) : result.mismatched_paths.length ? (
                                <details className="evaluation-mismatch-details">
                                  <summary>{result.mismatched_paths.length} paths</summary>
                                  <div>{result.mismatched_paths.map((path) => <code key={path}>{path}</code>)}</div>
                                </details>
                              ) : (
                                <span className="evaluation-no-mismatch">None</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </>
          )}
        </section>
      </div>

      {showCreate && (
        <div className="evaluation-modal-backdrop" role="presentation">
          <section
            className="evaluation-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="evaluation-create-title"
          >
            <div className="evaluation-modal-header">
              <div>
                <h3 id="evaluation-create-title">New evaluation</h3>
              </div>
              <button
                type="button"
                className="evaluation-close-button"
                onClick={closeCreate}
                aria-label="Close evaluation setup"
                disabled={creating}
              >
                ×
              </button>
            </div>

            <div className="evaluation-modal-body">
              {discoveryLoading && catalogs.length === 0 ? (
                <div className="evaluation-setup-state" role="status">
                  <span className="spinner" aria-hidden="true" />
                  <strong>Loading Unity Catalog…</strong>
                </div>
              ) : discoveryBlocked ? (
                <div className="evaluation-setup-state is-error" role="alert">
                  <strong>Unity Catalog access is unavailable</strong>
                  <p>{setupError}</p>
                  <div>
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => openCreate().catch(() => undefined)}
                    >
                      Retry
                    </button>
                    <button type="button" className="btn-submit" onClick={closeCreate}>
                      Close
                    </button>
                  </div>
                </div>
              ) : (
                <>
              {setupError && (
                <div className="evaluation-alert is-error" role="alert">
                  <strong>Unable to continue</strong>
                  <span>{setupError}</span>
                </div>
              )}

              <section className="evaluation-form-section">
                <div className="evaluation-form-section-heading">
                  <span>1</span>
                  <strong>Dataset</strong>
                </div>
                <div className="evaluation-form-grid">
                  <label className="evaluation-field full-width">
                    <span>Run name</span>
                    <input
                      value={draft.name}
                      onChange={(event) => updateDraft("name", event.target.value)}
                      maxLength={120}
                    />
                  </label>
                  <label className="evaluation-field">
                    <span>Catalog</span>
                    <select
                      value={draft.catalog}
                      onChange={(event) => handleCatalogChange(event.target.value)}
                      disabled={discoveryLoading}
                    >
                      <option value="">Select catalog…</option>
                      {catalogs.map((catalog) => <option key={catalog}>{catalog}</option>)}
                    </select>
                  </label>
                  <label className="evaluation-field">
                    <span>Schema</span>
                    <select
                      value={draft.schema}
                      onChange={(event) => handleSchemaChange(event.target.value)}
                      disabled={!draft.catalog || discoveryLoading}
                    >
                      <option value="">Select schema…</option>
                      {schemas.map((schema) => <option key={schema}>{schema}</option>)}
                    </select>
                  </label>
                  <label className="evaluation-field full-width">
                    <span>UC table</span>
                    <select
                      value={draft.tableName}
                      onChange={(event) => handleTableChange(event.target.value)}
                      disabled={!draft.schema || discoveryLoading}
                    >
                      <option value="">Select table…</option>
                      {tables.map((table) => <option key={table}>{table}</option>)}
                    </select>
                  </label>
                  <label className="evaluation-field">
                    <span>Clinical text column</span>
                    <select
                      value={draft.inputColumn}
                      onChange={(event) => updateDraft("inputColumn", event.target.value)}
                      disabled={!draft.tableName || discoveryLoading}
                    >
                      <option value="">Select column…</option>
                      {columns.map((column) => (
                        <option key={column.name} value={column.name}>{column.name} · {column.data_type}</option>
                      ))}
                    </select>
                  </label>
                  <label className="evaluation-field">
                    <span>JSON ground truth column</span>
                    <select
                      value={draft.groundTruthColumn}
                      onChange={(event) => updateDraft("groundTruthColumn", event.target.value)}
                      disabled={!draft.tableName || discoveryLoading}
                    >
                      <option value="">Select column…</option>
                      {columns.map((column) => (
                        <option key={column.name} value={column.name}>{column.name} · {column.data_type}</option>
                      ))}
                    </select>
                  </label>
                  <label className="evaluation-field">
                    <span>Record ID column <small>Optional</small></span>
                    <select
                      value={draft.identifierColumn}
                      onChange={(event) => updateDraft("identifierColumn", event.target.value)}
                      disabled={!draft.tableName || discoveryLoading}
                    >
                      <option value="">Generated row number</option>
                      {columns.map((column) => (
                        <option key={column.name} value={column.name}>{column.name}</option>
                      ))}
                    </select>
                  </label>
                  <label className="evaluation-field">
                    <span>Rows</span>
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={draft.rowLimit}
                      onChange={(event) => updateDraft("rowLimit", event.target.value)}
                    />
                  </label>
                </div>
              </section>

              <section className="evaluation-form-section">
                <div className="evaluation-form-section-heading">
                  <span>2</span>
                  <strong>Model and preset</strong>
                </div>
                <div className="evaluation-form-grid">
                  <label className="evaluation-field">
                    <span>Model</span>
                    <select
                      value={draft.model}
                      onChange={(event) => updateDraft("model", event.target.value)}
                    >
                      <option value="">Select model…</option>
                      {evaluationModels.map((model) => (
                        <option
                          key={model.name}
                          value={model.name}
                          disabled={model.availability_status !== "ready"}
                        >
                          {model.display_name || model.name}{model.availability_status !== "ready" ? " · unavailable" : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="evaluation-field">
                    <span>Preset</span>
                    <select
                      value={draft.presetId}
                      onChange={(event) => updateDraft("presetId", event.target.value)}
                    >
                      <option value="">Select preset…</option>
                      {presets.map((preset) => (
                        <option key={preset.id} value={preset.id}>{preset.name}</option>
                      ))}
                    </select>
                  </label>
                </div>
                {selectedPreset && (
                  <div className="evaluation-preset-summary">
                    <strong>{selectedPreset.name}</strong>
                    <p>{selectedPreset.description}</p>
                  </div>
                )}
              </section>

              <details className="evaluation-judge-settings">
                <summary>
                  <span><strong>Optional MLflow judges</strong><small>Disabled by default; judges add latency and cost.</small></span>
                  <span>Advanced</span>
                </summary>
                <div>
                  <label>
                    <input
                      type="checkbox"
                      checked={draft.enableGroundedness}
                      onChange={(event) => updateDraft("enableGroundedness", event.target.checked)}
                    />
                    <span><strong>Groundedness</strong><small>Checks that extracted values are supported by source text.</small></span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={draft.enableCompleteness}
                      onChange={(event) => updateDraft("enableCompleteness", event.target.checked)}
                    />
                    <span><strong>Completeness</strong><small>Checks whether material ground-truth facts were captured.</small></span>
                  </label>
                </div>
              </details>

              <section className="evaluation-preflight">
                <div className="evaluation-preflight-heading">
                  <div>
                    <strong>Validate dataset</strong>
                    <p>Checks ground-truth JSON before inference.</p>
                  </div>
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={handlePreflight}
                    disabled={!datasetRequest || preflightLoading}
                  >
                    {preflightLoading ? "Validating…" : "Validate dataset"}
                  </button>
                </div>
                {preflight && (
                  <div className={`evaluation-preflight-result ${preflight.invalid_rows ? "is-invalid" : "is-valid"}`}>
                    <div>
                      <strong>{preflight.invalid_rows ? "Validation found issues" : "Ready to evaluate"}</strong>
                      <span>{preflight.valid_rows} valid · {preflight.invalid_rows} invalid · {preflight.row_count} rows checked</span>
                    </div>
                    {preflight.warnings.length > 0 && (
                      <ul>{preflight.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
                    )}
                    {preflight.samples.length > 0 && (
                      <details>
                        <summary>Preview {preflight.samples.length} sample rows</summary>
                        <div className="evaluation-sample-list">
                          {preflight.samples.map((sample) => (
                            <article key={sample.record_id}>
                              <strong>{sample.record_id}</strong>
                              <p>{sample.input_preview}</p>
                              <pre>{JSON.stringify(sample.ground_truth, null, 2)}</pre>
                            </article>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                )}
              </section>
                </>
              )}
            </div>

            {!discoveryLoading && !discoveryBlocked && (
              <div className="evaluation-modal-footer">
                <div>
                  <button type="button" className="btn-secondary" onClick={closeCreate} disabled={creating}>Cancel</button>
                  <button
                    type="button"
                    className="btn-submit"
                    onClick={handleCreate}
                    disabled={
                      creating ||
                      !draft.name.trim() ||
                      !draft.model ||
                      !draft.presetId ||
                      !preflight ||
                      preflight.invalid_rows > 0 ||
                      preflight.valid_rows === 0
                    }
                  >
                    {creating ? "Starting evaluation…" : "Run evaluation"}
                  </button>
                </div>
              </div>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
