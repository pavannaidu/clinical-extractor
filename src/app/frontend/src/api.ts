import type {
  Preset,
  PresetCreate,
  ModelInfo,
  ModelWarmupResponse,
  ModelSelectionEntry,
  ModelSettings,
  WorkspaceEndpoint,
  InferenceRequest,
  InferenceResponse,
  LoadTestRequest,
  LoadTestRun,
  LoadTestSampleResponse,
  LoadTestOutputEstimateRequest,
  LoadTestOutputEstimateResponse,
  ApiError,
  EvaluationColumn,
  EvaluationCreateRequest,
  EvaluationDatasetRequest,
  EvaluationPreflightResponse,
  EvaluationReport,
  EvaluationRun,
} from "./types";

const BASE = "/api/v1";

function parseApiError(status: number, body: unknown): ApiError {
  // FastAPI wraps HTTPException detail in { detail: ... }
  if (typeof body === "object" && body !== null) {
    const obj = body as Record<string, unknown>;
    // Structured error from our ErrorResponse
    if (obj.error_code && obj.message) {
      return obj as unknown as ApiError;
    }
    // FastAPI/Pydantic validation errors use { detail: [{ loc, msg, ... }] }.
    if (Array.isArray(obj.detail)) {
      const messages = obj.detail.flatMap((entry) => {
        if (typeof entry !== "object" || entry === null) return [];
        const issue = entry as Record<string, unknown>;
        if (typeof issue.msg !== "string") return [];
        const field = Array.isArray(issue.loc)
          ? issue.loc
              .filter((part) => part !== "body")
              .map(String)
              .join(".")
          : "";
        return [field ? `${field}: ${issue.msg}` : issue.msg];
      });
      if (messages.length > 0) {
        return {
          error_code: `HTTP_${status}`,
          message: messages.length === 1 ? messages[0] : "Request validation failed.",
          detail: messages.length > 1 ? messages.join(" · ") : undefined,
        };
      }
    }
    // FastAPI's { detail: { error_code, message, ... } }
    if (typeof obj.detail === "object" && obj.detail !== null) {
      const d = obj.detail as Record<string, unknown>;
      if (d.error_code && d.message) {
        return d as unknown as ApiError;
      }
    }
    // FastAPI's { detail: "string message" }
    if (typeof obj.detail === "string") {
      return {
        error_code: `HTTP_${status}`,
        message: obj.detail,
      };
    }
  }
  return {
    error_code: `HTTP_${status}`,
    message: "An unexpected error occurred.",
  };
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      body = await res.text().catch(() => null);
    }
    throw parseApiError(res.status, body);
  }
  return res.json();
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE}/models`);
  return handleResponse<ModelInfo[]>(res);
}

export async function fetchPresets(): Promise<Preset[]> {
  const res = await fetch(`${BASE}/presets`);
  return handleResponse<Preset[]>(res);
}

export async function createPreset(data: PresetCreate): Promise<Preset> {
  const res = await fetch(`${BASE}/presets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse<Preset>(res);
}

export async function updatePreset(
  id: string,
  data: Partial<PresetCreate>
): Promise<Preset> {
  const res = await fetch(`${BASE}/presets/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse<Preset>(res);
}

export async function deletePreset(id: string): Promise<void> {
  const res = await fetch(`${BASE}/presets/${id}`, { method: "DELETE" });
  if (!res.ok) {
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      body = null;
    }
    throw parseApiError(res.status, body);
  }
}

export async function runInference(
  req: InferenceRequest
): Promise<InferenceResponse> {
  const res = await fetch(`${BASE}/inference`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return handleResponse<InferenceResponse>(res);
}

export async function startLoadTest(req: LoadTestRequest): Promise<LoadTestRun> {
  const res = await fetch(`${BASE}/load-test/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return handleResponse<LoadTestRun>(res);
}

export async function fetchLoadTestRun(runId: string): Promise<LoadTestRun> {
  const res = await fetch(`${BASE}/load-test/runs/${encodeURIComponent(runId)}`);
  return handleResponse<LoadTestRun>(res);
}

export async function fetchActiveLoadTestRuns(): Promise<LoadTestRun[]> {
  const res = await fetch(`${BASE}/load-test/runs/active`);
  return handleResponse<LoadTestRun[]>(res);
}

export async function cancelLoadTestRun(runId: string): Promise<LoadTestRun> {
  const res = await fetch(`${BASE}/load-test/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  return handleResponse<LoadTestRun>(res);
}

export async function fetchLoadTestSample(): Promise<LoadTestSampleResponse> {
  const res = await fetch(`${BASE}/load-test/sample`);
  return handleResponse<LoadTestSampleResponse>(res);
}

export async function fetchLoadTestOutputEstimate(
  req: LoadTestOutputEstimateRequest
): Promise<LoadTestOutputEstimateResponse> {
  const res = await fetch(`${BASE}/load-test/output-estimate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return handleResponse<LoadTestOutputEstimateResponse>(res);
}

export async function calibrateLoadTestOutputEstimate(
  req: LoadTestOutputEstimateRequest
): Promise<LoadTestOutputEstimateResponse> {
  const res = await fetch(`${BASE}/load-test/output-estimate/calibrate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return handleResponse<LoadTestOutputEstimateResponse>(res);
}

export async function fetchWorkspaceEndpoints(): Promise<WorkspaceEndpoint[]> {
  const res = await fetch(`${BASE}/workspace/endpoints`);
  return handleResponse<WorkspaceEndpoint[]>(res);
}

export async function warmWorkspaceEndpoint(
  endpointName: string
): Promise<ModelWarmupResponse> {
  const res = await fetch(
    `${BASE}/workspace/endpoints/${encodeURIComponent(endpointName)}/warmup`,
    { method: "POST" }
  );
  return handleResponse<ModelWarmupResponse>(res);
}

export async function fetchModelSettings(): Promise<ModelSettings> {
  const res = await fetch(`${BASE}/settings/models`);
  return handleResponse<ModelSettings>(res);
}

export async function updateModelSettings(
  endpoints: ModelSelectionEntry[]
): Promise<ModelSettings> {
  const res = await fetch(`${BASE}/settings/models`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ endpoints }),
  });
  return handleResponse<ModelSettings>(res);
}

export async function fetchEvaluationCatalogs(): Promise<string[]> {
  const res = await fetch(`${BASE}/evaluations/catalogs`);
  const body = await handleResponse<{ items: string[] }>(res);
  return body.items;
}

export async function fetchEvaluationSchemas(catalog: string): Promise<string[]> {
  const res = await fetch(
    `${BASE}/evaluations/schemas?catalog=${encodeURIComponent(catalog)}`
  );
  const body = await handleResponse<{ items: string[] }>(res);
  return body.items;
}

export async function fetchEvaluationTables(
  catalog: string,
  schema: string
): Promise<string[]> {
  const params = new URLSearchParams({ catalog, schema });
  const res = await fetch(`${BASE}/evaluations/tables?${params}`);
  const body = await handleResponse<{ items: string[] }>(res);
  return body.items;
}

export async function fetchEvaluationColumns(
  tableName: string
): Promise<EvaluationColumn[]> {
  const res = await fetch(
    `${BASE}/evaluations/columns?table_name=${encodeURIComponent(tableName)}`
  );
  const body = await handleResponse<{ columns: EvaluationColumn[] }>(res);
  return body.columns;
}

export async function preflightEvaluation(
  req: EvaluationDatasetRequest
): Promise<EvaluationPreflightResponse> {
  const res = await fetch(`${BASE}/evaluations/preflight`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return handleResponse<EvaluationPreflightResponse>(res);
}

export async function fetchEvaluationRuns(): Promise<EvaluationRun[]> {
  const res = await fetch(`${BASE}/evaluations/runs`);
  return handleResponse<EvaluationRun[]>(res);
}

export async function createEvaluationRun(
  req: EvaluationCreateRequest
): Promise<EvaluationRun> {
  const res = await fetch(`${BASE}/evaluations/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return handleResponse<EvaluationRun>(res);
}

export async function fetchEvaluationRun(runId: string): Promise<EvaluationRun> {
  const res = await fetch(`${BASE}/evaluations/runs/${encodeURIComponent(runId)}`);
  return handleResponse<EvaluationRun>(res);
}

export async function fetchEvaluationReport(
  runId: string
): Promise<EvaluationReport> {
  const res = await fetch(
    `${BASE}/evaluations/runs/${encodeURIComponent(runId)}/report`
  );
  return handleResponse<EvaluationReport>(res);
}

export async function cancelEvaluationRun(runId: string): Promise<EvaluationRun> {
  const res = await fetch(`${BASE}/evaluations/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  return handleResponse<EvaluationRun>(res);
}

export async function rerunEvaluation(runId: string): Promise<EvaluationRun> {
  const res = await fetch(
    `${BASE}/evaluations/runs/${encodeURIComponent(runId)}/rerun`,
    { method: "POST" }
  );
  return handleResponse<EvaluationRun>(res);
}
