export interface Preset {
  id: string;
  name: string;
  description: string;
  schema_json: string;
  example_prompt: string;
  system_prompt: string;
  is_builtin: boolean;
}

export interface PresetCreate {
  name: string;
  description: string;
  schema_json: string;
  example_prompt: string;
  system_prompt: string;
}

export interface ModelInfo {
  name: string;
  display_name?: string | null;
  model_type: "custom" | "fmapi" | "ai_extract";
  max_tokens: number | null;
  serving_runtime: "vllm_beta" | "standard" | null;
  availability_status: "ready" | "unavailable";
  availability_message: string | null;
  rate_limits?: ModelRateLimits | null;
}

export interface ModelRateLimits {
  docs_model_name: string;
  itpm?: number | null;
  otpm?: number | null;
  qph?: number | null;
  docs_url: string;
  docs_updated_on: string;
}

export interface InferenceRequest {
  prompt: string;
  schema_json: string;
  system_prompt: string;
  model: string;
  runtime?: "model_serving" | "ai_extract";
  ai_extract_options?: {
    precision: boolean;
    enable_citations: boolean;
    enable_confidence_scores: boolean;
  } | null;
  max_tokens?: number | null;
}

export interface ExtractionCitationBox {
  coord: number[];
  page_id: number;
}

export interface ExtractionCitation {
  id: number;
  start?: number | null;
  stop?: number | null;
  bbox?: ExtractionCitationBox[] | null;
}

export interface ExtractionEvidenceField {
  path: string;
  value: unknown;
  confidence_score?: number | null;
  citation_ids: number[];
}

export interface ExtractionEvidence {
  version?: string | null;
  mode?: string | null;
  chunk_type?: string | null;
  fields: ExtractionEvidenceField[];
  citations: ExtractionCitation[];
}

export interface InferenceResponse {
  prediction: unknown;
  latency_ms: number;
  raw_output: string;
  model: string;
  runtime?: "model_serving" | "ai_extract";
  schema_warnings?: string[];
  evidence?: ExtractionEvidence | null;
  finish_reason?: string | null;
  truncated: boolean;
  repaired_output: boolean;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  output_tokens_per_second?: number | null;
}

export interface LoadTestRequest {
  model: string;
  total_requests: number;
  max_workers: number;
  max_tokens: number;
  scheduling_mode?: "burst" | "target_rate";
  duration_s?: number;
  load_multiplier?: number;
  expected_output_tokens?: number | null;
  use_documented_targets?: boolean;
  target_itpm?: number | null;
  target_otpm?: number | null;
  prompt?: string | null;
  schema_json?: string | null;
  system_prompt?: string | null;
}

export interface LoadTestConfig {
  total_requests: number;
  max_workers: number;
  max_retries: number;
  max_tokens: number;
  cache_bust: boolean;
  target_source: "documented" | "custom" | "none";
  use_documented_targets?: boolean;
  target_itpm?: number | null;
  target_otpm?: number | null;
  target_itpm_source?: "documented" | "custom" | "none";
  target_otpm_source?: "documented" | "custom" | "none";
  scheduling_mode?: "burst" | "target_rate";
  duration_s?: number | null;
  load_multiplier?: number;
  pacing_mode?: "token_target" | "burst";
  input_token_estimation?: "serialized_payload_bytes_div_4";
  estimated_input_tokens_per_request?: number;
  planned_output_tokens_per_request?: number;
  request_interval_s?: number;
  planned_requests_per_minute?: number | null;
  planned_itpm?: number | null;
  planned_otpm?: number | null;
  planned_duration_s?: number;
  limiting_target?: "itpm" | "otpm" | "both" | "none";
}

export interface LoadTestSummary {
  succeeded: number;
  failed: number;
  retried: number;
  retried_succeeded: number;
  retry_exhausted: number;
  total_attempts: number;
  wall_time_s: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  usage_reported: number;
  usage_missing: number;
  usage_coverage: number;
  achieved_itpm: number | null;
  achieved_otpm: number | null;
  target_itpm?: number | null;
  target_otpm?: number | null;
  planned_itpm?: number | null;
  planned_otpm?: number | null;
  achieved_requests_per_minute?: number | null;
  successful_requests_per_minute?: number | null;
  itpm_target_achievement_ratio?: number | null;
  otpm_target_achievement_ratio?: number | null;
  pacing_slots_used?: number;
  average_start_delay_s?: number;
  max_start_delay_s?: number;
  actual_last_start_s?: number;
  p50_latency_s?: number | null;
  p95_latency_s?: number | null;
  p99_latency_s?: number | null;
  peak_rolling_itpm?: number | null;
  peak_rolling_otpm?: number | null;
}

export interface LoadTestSeriesPoint {
  elapsed_s: number;
  cumulative_prompt_tokens: number;
  cumulative_completion_tokens: number;
  target_cumulative_prompt_tokens?: number | null;
  target_cumulative_completion_tokens?: number | null;
  target_cumulative_tokens?: number | null;
  rolling_itpm?: number | null;
  rolling_otpm?: number | null;
}

export interface LoadTestRequestResult {
  req_id: number;
  status: string;
  attempts: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  response_completion_tokens?: number | null;
  elapsed_s: number;
  abs_end_s: number;
  error?: string | null;
  retry_events?: Array<{
    attempt: number;
    elapsed_s: number;
    status_code?: number | null;
    error_code?: string | null;
    retry_after_s?: number | null;
    backoff_s: number;
    prompt_tokens?: number | null;
    completion_tokens?: number | null;
  }>;
  scheduled_start_s?: number;
  actual_start_s?: number;
  start_delay_s?: number;
  pacing_wait_s?: number;
  last_attempt_start_s?: number;
}

export interface LoadTestResponse {
  model: string;
  config: LoadTestConfig;
  summary: LoadTestSummary;
  series: LoadTestSeriesPoint[];
  requests: LoadTestRequestResult[];
}

export interface LoadTestSampleResponse {
  name: string;
  description: string;
  prompt: string;
  schema_json: string;
  system_prompt: string;
}

export interface LoadTestOutputEstimateRequest {
  model: string;
  max_tokens: number;
  prompt: string;
  schema_json: string;
  system_prompt: string;
}

export interface LoadTestOutputEstimateResponse {
  model: string;
  workload_key: string;
  tokens: number | null;
  sample_count: number;
  source: "calibration" | "load_test" | "none";
  measured_at: string | null;
  persistence: "lakebase" | "memory" | "none";
}

export type LoadTestRunStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface LoadTestRunProgress {
  total_requests: number;
  completed_requests: number;
  succeeded: number;
  failed: number;
  total_attempts: number;
}

export interface LoadTestRun {
  run_id: string;
  model: string;
  status: LoadTestRunStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress: LoadTestRunProgress;
  config: LoadTestConfig | null;
  partial_result: LoadTestResponse | null;
  result: LoadTestResponse | null;
  error: ApiError | null;
}

export interface WorkspaceEndpoint {
  name: string;
  model_type: "custom" | "fmapi";
  ready: boolean;
  serving_runtime: "vllm_beta" | "standard" | null;
  selected: boolean;
}

export interface ModelWarmupResponse {
  name: string;
  status: "ready" | "warming";
  message: string;
}

export interface ModelSelectionEntry {
  name: string;
  model_type: "custom" | "fmapi";
  serving_runtime?: "vllm_beta" | "standard" | null;
}

export interface ModelSettings {
  endpoints: ModelSelectionEntry[];
  source: "saved" | "memory" | "default";
}

export interface ApiError {
  error_code: string;
  message: string;
  detail?: string;
}

export interface ModelSlot {
  id: string;
  selectedModel: string;
  loading: boolean;
  result: InferenceResponse | null;
  error: ApiError | null;
  notice: string | null;
  activeTab: "formatted" | "raw" | "evidence";
  aiExtractPrecision: boolean;
  aiExtractCitations: boolean;
  aiExtractConfidenceScores: boolean;
  maxOutputTokensOverride: string;
  copied: boolean;
  outputTokenLimit: number | null;
  outputTokenLimitSource: "model_default" | "override" | null;
}
