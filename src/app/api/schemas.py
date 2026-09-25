import json
from datetime import datetime

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, Literal, Optional


def _validate_json_schema_text(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict) or "type" not in parsed:
        raise ValueError('Schema must be a JSON object with a "type" key')

    try:
        Draft202012Validator.check_schema(parsed)
    except SchemaError as exc:
        raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc
    return value


class ModelRateLimits(BaseModel):
    docs_model_name: str
    itpm: int | None = None
    otpm: int | None = None
    qph: int | None = None
    docs_url: str
    docs_updated_on: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    display_name: str | None = None
    model_type: Literal["custom", "fmapi", "ai_extract"]
    max_tokens: int | None
    serving_runtime: Literal["vllm_beta", "standard"] | None = None
    availability_status: Literal["ready", "unavailable"]
    availability_message: str | None = None
    rate_limits: ModelRateLimits | None = None


class AiExtractOptions(BaseModel):
    precision: bool = False


class InferenceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    prompt: str = Field(..., max_length=10000)
    json_schema: str = Field(..., alias="schema_json", max_length=20000)
    system_prompt: str = Field(default="", max_length=4000)
    model: str = Field(default="", max_length=200)
    runtime: Literal["model_serving", "ai_extract"] = "model_serving"
    ai_extract_options: AiExtractOptions | None = None
    # Optional per-request override of the output token cap. When omitted, the
    # per-endpoint default from models.json (settings) is used.
    max_tokens: int | None = Field(default=None, ge=1, le=8192)

    @field_validator("json_schema")
    @classmethod
    def validate_schema_json(cls, v: str) -> str:
        return _validate_json_schema_text(v)


class ExtractionCitationBox(BaseModel):
    coord: list[int] = Field(default_factory=list)
    page_id: int


class ExtractionCitation(BaseModel):
    id: int
    start: int | None = None
    stop: int | None = None
    bbox: list[ExtractionCitationBox] | None = None


class ExtractionEvidenceField(BaseModel):
    path: str
    value: Any = None
    confidence_score: float | None = None
    citation_ids: list[int] = Field(default_factory=list)


class ExtractionEvidence(BaseModel):
    version: str | None = None
    mode: str | None = None
    chunk_type: str | None = None
    fields: list[ExtractionEvidenceField] = Field(default_factory=list)
    citations: list[ExtractionCitation] = Field(default_factory=list)


class InferenceResponse(BaseModel):
    prediction: Any
    latency_ms: float
    raw_output: str
    model: str = ""
    runtime: Literal["model_serving", "ai_extract"] = "model_serving"
    schema_warnings: list[str] = Field(default_factory=list)
    evidence: ExtractionEvidence | None = None
    finish_reason: str | None = None
    truncated: bool = False
    repaired_output: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    output_tokens_per_second: float | None = None


class LoadTestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    model: str = Field(..., max_length=200)
    total_requests: int = Field(default=150, ge=1, le=150)
    max_workers: int = Field(default=10, ge=1, le=10)
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=4096,
        description="Per-invocation output cap and conservative output-token pacing budget.",
    )
    scheduling_mode: Literal["burst", "paced", "target_rate"] = Field(
        default="burst",
        description=(
            "Burst admits logical requests immediately behind the worker limit. "
            "Target rate schedules arrivals by wall clock using the configured token targets. "
            "Paced is retained as a backwards-compatible alias."
        ),
    )
    duration_s: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Target-rate arrival window. Ignored for burst tests.",
    )
    load_multiplier: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Fraction or multiple of the user-verified token targets to schedule.",
    )
    expected_output_tokens: int | None = Field(
        default=None,
        ge=1,
        le=4096,
        description="Planning estimate per response; defaults to max_tokens.",
    )
    target_itpm: int | None = Field(
        default=None,
        ge=1,
        le=10_000_000,
        description=(
            "User-verified input tokens per minute reference."
        ),
    )
    target_otpm: int | None = Field(
        default=None,
        ge=1,
        le=10_000_000,
        description=(
            "User-verified output tokens per minute reference."
        ),
    )
    use_documented_targets: bool = Field(
        default=False,
        description=(
            "Legacy API option that fills omitted axes from endpoint metadata. The UI "
            "always sends false so endpoint selection never changes user references."
        ),
    )
    prompt: str | None = Field(
        default=None,
        max_length=100000,
        description="Custom workload prompt; null uses the built-in clinical sample.",
    )
    json_schema: str | None = Field(
        default=None,
        alias="schema_json",
        max_length=20000,
        description="Custom JSON Schema; null uses the built-in clinical schema.",
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=4000,
        description=(
            "Custom system prompt; null uses the built-in instruction and an empty "
            "string intentionally omits it."
        ),
    )

    @field_validator("json_schema")
    @classmethod
    def validate_schema_json(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_json_schema_text(value)


class LoadTestConfig(BaseModel):
    total_requests: int
    max_workers: int
    max_retries: int
    max_tokens: int
    cache_bust: bool
    target_source: Literal["documented", "custom", "none"]
    use_documented_targets: bool = False
    target_itpm: int | None = None
    target_otpm: int | None = None
    target_itpm_source: Literal["documented", "custom", "none"] = "none"
    target_otpm_source: Literal["documented", "custom", "none"] = "none"
    scheduling_mode: Literal["burst", "target_rate"] = "burst"
    duration_s: int | None = None
    load_multiplier: float = 1.0
    pacing_mode: Literal["token_target", "burst"] = "burst"
    input_token_estimation: Literal["serialized_payload_bytes_div_4"] = (
        "serialized_payload_bytes_div_4"
    )
    estimated_input_tokens_per_request: int = 0
    planned_output_tokens_per_request: int = 0
    request_interval_s: float = Field(
        default=0.0,
        description=(
            "Wall-clock interval between initial arrivals. Retries re-enter after backoff. "
            "It is max(estimated input tokens / ITPM, planned output tokens / OTPM) * 60."
        ),
    )
    planned_requests_per_minute: float | None = None
    planned_itpm: float | None = None
    planned_otpm: float | None = None
    planned_duration_s: float = Field(
        default=0.0,
        description="Planned first-to-last initial request launch window; excludes inference latency.",
    )
    limiting_target: Literal["itpm", "otpm", "both", "none"] = "none"


class LoadTestSummary(BaseModel):
    succeeded: int
    failed: int
    retried: int = Field(
        description="Logical requests that made more than one endpoint invocation."
    )
    retried_succeeded: int = Field(
        description="Retried logical requests that eventually succeeded."
    )
    retry_exhausted: int = Field(
        description="Logical requests that failed after consuming every configured retry."
    )
    total_attempts: int = Field(
        description="Total endpoint invocations across all logical requests."
    )
    wall_time_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    usage_reported: int = Field(
        description="Logical request results containing both prompt and completion token counts."
    )
    usage_missing: int = Field(
        description="Logical request results missing either prompt or completion token counts."
    )
    usage_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of logical request results with complete token usage data.",
    )
    achieved_itpm: float | None = Field(
        description=(
            "All response-reported prompt tokens divided by total wall time. Null unless "
            "every logical result reports complete usage."
        )
    )
    achieved_otpm: float | None = Field(
        description=(
            "All response-reported completion tokens divided by total wall time. Null unless "
            "every logical result reports complete usage."
        )
    )
    target_itpm: int | None = None
    target_otpm: int | None = None
    planned_itpm: float | None = None
    planned_otpm: float | None = None
    achieved_requests_per_minute: float | None = None
    successful_requests_per_minute: float | None = None
    itpm_target_achievement_ratio: float | None = None
    otpm_target_achievement_ratio: float | None = None
    pacing_slots_used: int = 0
    average_start_delay_s: float = 0.0
    max_start_delay_s: float = 0.0
    actual_last_start_s: float = 0.0
    p50_latency_s: float | None = None
    p95_latency_s: float | None = None
    p99_latency_s: float | None = None
    peak_rolling_itpm: float | None = None
    peak_rolling_otpm: float | None = None


class LoadTestSeriesPoint(BaseModel):
    elapsed_s: float
    cumulative_prompt_tokens: int
    cumulative_completion_tokens: int
    target_cumulative_prompt_tokens: float | None = None
    target_cumulative_completion_tokens: float | None = None
    target_cumulative_tokens: float | None = None
    rolling_itpm: float | None = None
    rolling_otpm: float | None = None


class LoadTestRetryEvent(BaseModel):
    attempt: int
    elapsed_s: float
    status_code: int | None = None
    error_code: str | None = None
    retry_after_s: float | None = None
    backoff_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LoadTestRequestResult(BaseModel):
    req_id: int
    status: str
    attempts: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_completion_tokens: int | None = Field(
        default=None,
        description="Output tokens from the successful response only; excludes retry attempts.",
    )
    elapsed_s: float
    abs_end_s: float
    error: str | None = None
    retry_events: list[LoadTestRetryEvent] = Field(default_factory=list)
    scheduled_start_s: float = 0.0
    actual_start_s: float = 0.0
    start_delay_s: float = 0.0
    pacing_wait_s: float = 0.0
    last_attempt_start_s: float = 0.0


class LoadTestResponse(BaseModel):
    model: str
    config: LoadTestConfig
    summary: LoadTestSummary
    series: list[LoadTestSeriesPoint]
    requests: list[LoadTestRequestResult]


class LoadTestPlanResponse(BaseModel):
    model: str
    config: LoadTestConfig
    rate_limits: ModelRateLimits | None = None
    runnable: bool
    max_planned_duration_s: float
    suggested_max_requests: int
    validation_message: str | None = None


class LoadTestRunProgress(BaseModel):
    total_requests: int
    completed_requests: int = 0
    succeeded: int = 0
    failed: int = 0
    total_attempts: int = 0


class LoadTestRunError(BaseModel):
    error_code: str
    message: str
    detail: str | None = None


class LoadTestRunResponse(BaseModel):
    run_id: str
    model: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    progress: LoadTestRunProgress
    config: LoadTestConfig
    partial_result: LoadTestResponse | None = None
    result: LoadTestResponse | None = None
    error: LoadTestRunError | None = None


class LoadTestSampleResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    name: str
    description: str
    prompt: str
    json_schema: str = Field(..., alias="schema_json")
    system_prompt: str


class LoadTestOutputEstimateRequest(BaseModel):
    """Workload identity used to find or measure output length."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    model: str = Field(..., max_length=200)
    max_tokens: int = Field(default=4096, ge=1, le=4096)
    prompt: str = Field(..., min_length=1, max_length=100000)
    json_schema: str = Field(..., alias="schema_json", max_length=20000)
    system_prompt: str = Field(default="", max_length=4000)

    @field_validator("json_schema")
    @classmethod
    def validate_schema_json(cls, value: str) -> str:
        return _validate_json_schema_text(value)


class LoadTestOutputEstimateResponse(BaseModel):
    model: str
    workload_key: str
    tokens: int | None = None
    sample_count: int = 0
    source: Literal["calibration", "load_test", "none"] = "none"
    measured_at: datetime | None = None
    persistence: Literal["lakebase", "memory", "none"] = "none"


class WorkspaceEndpoint(BaseModel):
    """A serving endpoint discovered in the workspace, for the settings pane."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    model_type: Literal["custom", "fmapi"]
    ready: bool
    serving_runtime: Literal["vllm_beta", "standard"] | None = None
    selected: bool = False


class ModelWarmupResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    status: Literal["ready", "warming"]
    message: str


class ModelSelectionEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=200)
    model_type: Literal["custom", "fmapi"]
    serving_runtime: Literal["vllm_beta", "standard"] | None = None


class ModelSettings(BaseModel):
    """The set of endpoints an operator has chosen to surface in the UI."""

    endpoints: list[ModelSelectionEntry]
    source: Literal["saved", "memory", "default"] = "default"


class ModelSettingsUpdate(BaseModel):
    endpoints: list[ModelSelectionEntry] = Field(..., min_length=1, max_length=500)


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: Optional[str] = None


class Preset(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    id: str
    name: str
    description: str
    json_schema: str = Field(..., alias="schema_json")
    example_prompt: str
    system_prompt: str = ""
    is_builtin: bool = False


class PresetCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)
    json_schema: str = Field(..., alias="schema_json", max_length=20000)
    example_prompt: str = Field("", max_length=10000)
    system_prompt: str = Field("", max_length=4000)

    @field_validator("json_schema")
    @classmethod
    def validate_schema_json(cls, v: str) -> str:
        return _validate_json_schema_text(v)


class PresetUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    json_schema: Optional[str] = Field(None, alias="schema_json", max_length=20000)
    example_prompt: Optional[str] = Field(None, max_length=10000)
    system_prompt: Optional[str] = Field(None, max_length=4000)

    @field_validator("json_schema")
    @classmethod
    def validate_schema_json(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return _validate_json_schema_text(v)
