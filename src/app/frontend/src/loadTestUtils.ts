import type {
  LoadTestRequest,
  LoadTestRequestResult,
  ModelInfo,
} from "./types";

export const LOAD_TEST_MAX_REQUESTS = 150;
export const LOAD_TEST_MAX_WORKERS = 10;
export const LOAD_TEST_MAX_TOKENS = 4096;
export const LOAD_TEST_MAX_TARGET_TPM = 10_000_000;
export const LOAD_TEST_PROMPT_MAX_LENGTH = 100_000;
export const LOAD_TEST_SCHEMA_MAX_LENGTH = 20_000;
export const LOAD_TEST_SYSTEM_PROMPT_MAX_LENGTH = 4_000;
export const DEFAULT_REFERENCE_ITPM = 200_000;
export const DEFAULT_REFERENCE_OTPM = 20_000;
export const DEFAULT_REFERENCE_QPH = 360_000;
export const LOAD_TEST_MIN_DURATION_S = 10;
export const LOAD_TEST_MAX_DURATION_S = 600;

export interface LoadTestFormValues {
  schedulingMode: "burst" | "target_rate";
  totalRequests: string;
  durationSeconds: string;
  loadMultiplier: string;
  expectedOutputTokens: string;
  maxWorkers: string;
  maxTokens: string;
  targetItpm: string;
  targetOtpm: string;
  prompt: string;
  schemaJson: string;
  systemPrompt: string;
}

export type LoadTestField = keyof LoadTestFormValues | "model";
export type LoadTestFieldErrors = Partial<Record<LoadTestField, string>>;

export function parseIntegerInput(value: string): number | null {
  const trimmed = value.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const parsed = Number(trimmed);
  return Number.isSafeInteger(parsed) ? parsed : null;
}

function parseDecimalInput(value: string): number | null {
  const trimmed = value.trim();
  if (!/^\d+(\.\d+)?$/.test(trimmed)) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function validateSchema(schemaJson: string): string | null {
  try {
    const schema = JSON.parse(schemaJson) as unknown;
    if (
      typeof schema !== "object" ||
      schema === null ||
      !("type" in schema)
    ) {
      return 'Enter a JSON Schema object with a "type" property.';
    }
  } catch (error) {
    return error instanceof Error ? `Invalid JSON: ${error.message}` : "Invalid JSON.";
  }
  return null;
}

export function buildLoadTestRequest(
  form: LoadTestFormValues,
  model: ModelInfo | null
): { request: LoadTestRequest | null; errors: LoadTestFieldErrors } {
  const errors: LoadTestFieldErrors = {};
  if (!model) errors.model = "Choose a ready configured endpoint.";

  const totalRequests = parseIntegerInput(form.totalRequests);
  if (
    form.schedulingMode === "burst" &&
    (totalRequests == null || totalRequests < 1 || totalRequests > LOAD_TEST_MAX_REQUESTS)
  ) {
    errors.totalRequests = `Enter 1–${LOAD_TEST_MAX_REQUESTS} requests.`;
  }

  const durationSeconds = parseIntegerInput(form.durationSeconds);
  if (
    form.schedulingMode === "target_rate" &&
    (durationSeconds == null ||
      durationSeconds < LOAD_TEST_MIN_DURATION_S ||
      durationSeconds > LOAD_TEST_MAX_DURATION_S)
  ) {
    errors.durationSeconds = `Enter ${LOAD_TEST_MIN_DURATION_S}–${LOAD_TEST_MAX_DURATION_S} seconds.`;
  }

  const loadMultiplier = parseDecimalInput(form.loadMultiplier);
  if (
    form.schedulingMode === "target_rate" &&
    (loadMultiplier == null || loadMultiplier < 0.1 || loadMultiplier > 3)
  ) {
    errors.loadMultiplier = "Enter 0.1–3×.";
  }

  const maxWorkers = parseIntegerInput(form.maxWorkers);
  if (maxWorkers == null || maxWorkers < 1 || maxWorkers > LOAD_TEST_MAX_WORKERS) {
    errors.maxWorkers = `Enter 1–${LOAD_TEST_MAX_WORKERS} workers.`;
  }

  const maxTokensCap = Math.min(model?.max_tokens ?? LOAD_TEST_MAX_TOKENS, LOAD_TEST_MAX_TOKENS);
  const maxTokens = parseIntegerInput(form.maxTokens);
  if (maxTokens == null || maxTokens < 1 || maxTokens > maxTokensCap) {
    errors.maxTokens = `Enter 1–${maxTokensCap} output tokens.`;
  }

  const parseTarget = (field: "targetItpm" | "targetOtpm", label: string) => {
    if (!form[field].trim()) return null;
    const value = parseIntegerInput(form[field]);
    if (value == null || value < 1 || value > LOAD_TEST_MAX_TARGET_TPM) {
      errors[field] = `${label} must be 1–${LOAD_TEST_MAX_TARGET_TPM.toLocaleString()}.`;
      return null;
    }
    return value;
  };
  const targetItpm = parseTarget("targetItpm", "ITPM reference");
  const targetOtpm = parseTarget("targetOtpm", "OTPM reference");
  if (
    form.schedulingMode === "target_rate" &&
    targetItpm == null &&
    targetOtpm == null &&
    !errors.targetItpm &&
    !errors.targetOtpm
  ) {
    errors.targetOtpm = "Enter an ITPM target, an OTPM target, or both.";
  }

  const expectedOutputTokens = parseIntegerInput(form.expectedOutputTokens);
  if (
    form.schedulingMode === "target_rate" &&
    targetOtpm != null &&
    (expectedOutputTokens == null || expectedOutputTokens < 1 || expectedOutputTokens > (maxTokens ?? maxTokensCap))
  ) {
    errors.expectedOutputTokens = "Calibrate output size or enter a manual estimate in Advanced.";
  }

  if (!form.prompt.trim()) errors.prompt = "Enter a workload prompt.";
  if (form.prompt.length > LOAD_TEST_PROMPT_MAX_LENGTH) {
    errors.prompt = `Prompt must be ${LOAD_TEST_PROMPT_MAX_LENGTH.toLocaleString()} characters or fewer.`;
  }
  if (!form.schemaJson.trim()) errors.schemaJson = "Enter a JSON response schema.";
  else if (form.schemaJson.length > LOAD_TEST_SCHEMA_MAX_LENGTH) {
    errors.schemaJson = `Schema must be ${LOAD_TEST_SCHEMA_MAX_LENGTH.toLocaleString()} characters or fewer.`;
  } else {
    const schemaError = validateSchema(form.schemaJson);
    if (schemaError) errors.schemaJson = schemaError;
  }
  if (form.systemPrompt.length > LOAD_TEST_SYSTEM_PROMPT_MAX_LENGTH) {
    errors.systemPrompt = `System prompt must be ${LOAD_TEST_SYSTEM_PROMPT_MAX_LENGTH.toLocaleString()} characters or fewer.`;
  }

  if (
    Object.keys(errors).length > 0 ||
    !model ||
    (form.schedulingMode === "burst" && totalRequests == null) ||
    (form.schedulingMode === "target_rate" &&
      (durationSeconds == null ||
        loadMultiplier == null ||
        (targetOtpm != null && expectedOutputTokens == null))) ||
    maxWorkers == null ||
    maxTokens == null
  ) {
    return { request: null, errors };
  }

  return {
    request: {
      model: model.name,
      total_requests: form.schedulingMode === "burst" ? (totalRequests ?? 1) : 1,
      max_workers: maxWorkers,
      max_tokens: maxTokens,
      scheduling_mode: form.schedulingMode,
      ...(form.schedulingMode === "target_rate"
        ? {
            duration_s: durationSeconds ?? 60,
            load_multiplier: loadMultiplier ?? 1,
            ...(expectedOutputTokens == null
              ? {}
              : { expected_output_tokens: expectedOutputTokens }),
          }
        : {}),
      use_documented_targets: false,
      ...(targetItpm == null ? {} : { target_itpm: targetItpm }),
      ...(targetOtpm == null ? {} : { target_otpm: targetOtpm }),
      prompt: form.prompt,
      schema_json: form.schemaJson,
      system_prompt: form.systemPrompt,
    },
    errors,
  };
}

export function percentile(values: number[], percentileValue: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.ceil((percentileValue / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(sorted.length - 1, index))];
}

export function requestLatencyPercentile(
  requests: LoadTestRequestResult[],
  percentileValue: number
): number | null {
  return percentile(
    requests.map((request) => request.elapsed_s).filter(Number.isFinite),
    percentileValue
  );
}
