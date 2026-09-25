import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator
from pydantic import ValidationError


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from api import routes  # noqa: E402
from api.schemas import (  # noqa: E402
    LoadTestOutputEstimateRequest,
    LoadTestRequest,
    ModelInfo,
)


CUSTOM_SCHEMA = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {"custom": {"type": "string"}},
        "required": ["custom"],
    }
)


class FakeAPIClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def do(self, method, path, body):
        self.calls.append({"method": method, "path": path, "body": body})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeWorkspaceClient:
    def __init__(self, responses=()):
        self.api_client = FakeAPIClient(responses)
        self.config = SimpleNamespace(host="https://workspace.example.com")


def _model(
    *,
    name="gpt-5-4-mini",
    model_type="fmapi",
    rate_limits=True,
):
    return ModelInfo(
        name=name,
        model_type=model_type,
        max_tokens=4096,
        serving_runtime="vllm_beta" if model_type == "custom" else None,
        availability_status="ready",
        availability_message=None,
        rate_limits=(routes._lookup_rate_limits(name) if rate_limits else None),
    )


def test_load_test_request_accepts_both_targets_and_custom_workload_alias():
    request = LoadTestRequest.model_validate(
        {
            "model": "gpt-5-4-mini",
            "target_itpm": 12_000,
            "target_otpm": 3_000,
            "prompt": "Extract this note",
            "schema_json": CUSTOM_SCHEMA,
            "system_prompt": "Return JSON",
        }
    )

    assert request.target_itpm == 12_000
    assert request.target_otpm == 3_000
    assert request.json_schema == CUSTOM_SCHEMA
    assert request.model_dump(by_alias=True)["schema_json"] == CUSTOM_SCHEMA

    with pytest.raises(ValidationError):
        LoadTestRequest(
            model="gpt-5-4-mini",
            schema_json='{"type": "not-a-json-schema-type"}',
        )


def test_load_test_request_accepts_large_prompts_up_to_100k_characters():
    request = LoadTestRequest(
        model="gpt-5-4-mini",
        prompt="x" * 100_000,
    )

    assert len(request.prompt or "") == 100_000

    with pytest.raises(ValidationError):
        LoadTestRequest(
            model="gpt-5-4-mini",
            prompt="x" * 100_001,
        )


def test_output_estimate_workload_key_canonicalizes_schema_formatting():
    compact = routes._load_test_workload_key(
        model="gpt-5-4-mini",
        max_tokens=1024,
        prompt="Extract this note",
        schema_json='{"type":"object","properties":{"x":{"type":"string"}}}',
        system_prompt="Return JSON",
    )
    formatted = routes._load_test_workload_key(
        model="gpt-5-4-mini",
        max_tokens=1024,
        prompt="Extract this note",
        schema_json=json.dumps(
            {"properties": {"x": {"type": "string"}}, "type": "object"},
            indent=2,
        ),
        system_prompt="Return JSON",
    )

    assert compact == formatted


def test_output_estimate_uses_median_and_memory_fallback(monkeypatch):
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)
    routes._OUTPUT_ESTIMATE_MEMORY.clear()

    record, persistence = routes._save_output_estimate(
        owner_id="user@example.com",
        workload_key="workload",
        completion_tokens=[100, 300, 200],
        source="load_test",
    )
    loaded, loaded_from = routes._get_output_estimate(
        "user@example.com", "workload"
    )

    assert record["tokens"] == 200
    assert record["sample_count"] == 3
    assert persistence == "memory"
    assert loaded == record
    assert loaded_from == "memory"


def test_calibrate_output_estimate_records_reported_usage(monkeypatch):
    request = Request({"type": "http", "headers": []})
    estimate_request = LoadTestOutputEstimateRequest(
        model="gpt-5-4-mini",
        max_tokens=128,
        prompt="Extract this note",
        schema_json=CUSTOM_SCHEMA,
        system_prompt="Return JSON",
    )
    monkeypatch.setattr(routes, "_load_allowed_endpoints", lambda: ["gpt-5-4-mini"])
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: object())
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)
    routes._OUTPUT_ESTIMATE_MEMORY.clear()

    async def calibrate(**_kwargs):
        return {
            "status": "ok",
            "response_completion_tokens": 321,
        }

    monkeypatch.setattr(routes, "_run_single_load_test_request", calibrate)
    response = asyncio.run(
        routes.calibrate_load_test_output_estimate(estimate_request, request)
    )

    assert response.tokens == 321
    assert response.sample_count == 1
    assert response.source == "calibration"
    assert response.persistence == "memory"


def test_load_test_transport_exposes_429_and_records_retry(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code, body, headers=None):
            self.status_code = status_code
            self._body = body
            self.headers = headers or {}

        def json(self):
            return self._body

    class FakeSession:
        def __init__(self):
            self.responses = [
                FakeResponse(
                    429,
                    {
                        "error_code": "REQUEST_LIMIT_EXCEEDED",
                        "message": "rate limited",
                    },
                    {"Retry-After": "7"},
                ),
                FakeResponse(
                    200,
                    {
                        "choices": [
                            {"message": {"content": '{"custom":"ok"}'}}
                        ],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 4},
                    },
                ),
            ]
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append({"url": url, **kwargs})
            return self.responses.pop(0)

    fake_session = FakeSession()
    monkeypatch.setattr(routes, "_load_test_http_session", lambda: fake_session)
    monkeypatch.setattr(routes.random, "uniform", lambda _a, _b: 1.25)

    async def no_sleep(_delay_s):
        return None

    monkeypatch.setattr(routes.asyncio, "sleep", no_sleep)
    workspace = SimpleNamespace(
        config=SimpleNamespace(
            host="https://workspace.example.com",
            user_agent="load-test",
            authenticate=lambda: {"Authorization": "Bearer token"},
        )
    )

    result = asyncio.run(
        routes._run_single_load_test_request(
            workspace_client=workspace,
            endpoint_name="gpt-5-4-mini",
            endpoint_settings={},
            req_id=2,
            max_tokens=128,
            max_retries=1,
            cache_bust=True,
            prompt="Custom prompt",
            schema_json=CUSTOM_SCHEMA,
            system_prompt="",
            schema_validator=Draft202012Validator(json.loads(CUSTOM_SCHEMA)),
        )
    )

    assert len(fake_session.calls) == 2
    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert result["retry_events"] == [
        {
            "attempt": 1,
            "elapsed_s": result["retry_events"][0]["elapsed_s"],
            "status_code": 429,
            "error_code": "REQUEST_LIMIT_EXCEEDED",
            "retry_after_s": 7.0,
            "backoff_s": 1.25,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
    ]


def test_rate_limit_detection_recognizes_sdk_exception_types():
    TooManyRequests = type("TooManyRequests", (Exception,), {})
    RequestLimitExceeded = type("RequestLimitExceeded", (Exception,), {})

    assert routes._is_rate_limit_error(TooManyRequests("throttled"))
    assert routes._is_rate_limit_error(RequestLimitExceeded("throttled"))


def test_resolve_load_test_targets_tracks_each_axis_source_independently():
    resolved = routes._resolve_load_test_target(
        model_info=_model(),
        target_itpm_override=75_000,
        target_otpm_override=None,
    )

    assert resolved == (75_000, 5_000, "custom", "custom", "documented")

    disabled_fallback = routes._resolve_load_test_target(
        model_info=_model(),
        target_itpm_override=None,
        target_otpm_override=5_000,
        use_documented_targets=False,
    )
    assert disabled_fallback == (
        None,
        5_000,
        "custom",
        "none",
        "custom",
    )

    custom_resolved = routes._resolve_load_test_target(
        model_info=_model(
            name="custom-endpoint",
            model_type="custom",
            rate_limits=False,
        ),
        target_itpm_override=None,
        target_otpm_override=8_000,
    )
    assert custom_resolved == (None, 8_000, "custom", "none", "custom")


def test_build_load_test_plan_uses_the_more_restrictive_token_axis():
    plan = routes._build_load_test_plan(
        total_requests=5,
        estimated_input_tokens_per_request=100,
        planned_output_tokens_per_request=200,
        target_itpm=6_000,
        target_otpm=6_000,
    )

    assert plan["pacing_mode"] == "token_target"
    assert plan["limiting_target"] == "otpm"
    assert plan["request_interval_s"] == pytest.approx(2.0)
    assert plan["planned_requests_per_minute"] == pytest.approx(30.0)
    assert plan["planned_itpm"] == pytest.approx(3_000.0)
    assert plan["planned_otpm"] == pytest.approx(6_000.0)
    assert plan["planned_duration_s"] == pytest.approx(8.0)


def test_build_load_test_plan_applies_target_load_multiplier():
    plan = routes._build_load_test_plan(
        total_requests=3,
        estimated_input_tokens_per_request=100,
        planned_output_tokens_per_request=200,
        target_itpm=6_000,
        target_otpm=6_000,
        scheduling_mode="target_rate",
        load_multiplier=1.25,
    )

    assert plan["request_interval_s"] == pytest.approx(1.6)
    assert plan["planned_otpm"] == pytest.approx(7_500.0)
    assert plan["limiting_target"] == "otpm"


def test_target_rate_derives_request_count_from_duration(monkeypatch):
    async def list_models():
        return [_model()]

    monkeypatch.setattr(routes, "list_models", list_models)
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})

    prepared = asyncio.run(
        routes._prepare_load_test(
            LoadTestRequest(
                model="gpt-5-4-mini",
                scheduling_mode="target_rate",
                duration_s=60,
                target_otpm=60,
                expected_output_tokens=60,
                max_tokens=128,
                use_documented_targets=False,
            ),
            max_planned_duration_s=900,
            max_wall_time_s=900,
        )
    )

    assert prepared["total_requests"] == 2
    assert prepared["config"].scheduling_mode == "target_rate"
    assert prepared["config"].duration_s == 60
    assert prepared["config"].planned_output_tokens_per_request == 60


def test_build_load_test_plan_without_targets_is_an_unpaced_burst():
    plan = routes._build_load_test_plan(
        total_requests=150,
        estimated_input_tokens_per_request=500,
        planned_output_tokens_per_request=1_024,
        target_itpm=None,
        target_otpm=None,
    )

    assert plan == {
        "pacing_mode": "burst",
        "request_interval_s": 0.0,
        "planned_requests_per_minute": None,
        "planned_itpm": None,
        "planned_otpm": None,
        "planned_duration_s": 0.0,
        "limiting_target": "none",
    }


def test_load_test_pacer_reserves_deterministic_slots(monkeypatch):
    clock = [100.0]

    async def advance_clock(delay_s):
        clock[0] += delay_s

    monkeypatch.setattr(routes.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(routes.asyncio, "sleep", advance_clock)

    async def exercise():
        pacer = routes._LoadTestPacer(request_interval_s=1.5, started_at=100.0)
        first = await pacer.wait_for_slot()
        second = await pacer.wait_for_slot()
        return pacer, first, second

    pacer, first, second = asyncio.run(exercise())

    assert first["scheduled_start_s"] == 0.0
    assert first["actual_start_s"] == 0.0
    assert second["scheduled_start_s"] == 1.5
    assert second["actual_start_s"] == 1.5
    assert second["pacing_wait_s"] == 1.5
    assert pacer.slots_used == 2


def test_load_test_pacer_does_not_catch_up_after_worker_delay(monkeypatch):
    clock = [50.0]

    async def advance_clock(delay_s):
        clock[0] += delay_s

    monkeypatch.setattr(routes.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(routes.asyncio, "sleep", advance_clock)

    async def exercise():
        pacer = routes._LoadTestPacer(request_interval_s=1.0, started_at=50.0)
        first = await pacer.wait_until(0.0)
        clock[0] = 53.0  # Simulate a saturated worker becoming available late.
        delayed_second = await pacer.wait_until(1.0)
        third = await pacer.wait_until(2.0)
        return first, delayed_second, third

    first, delayed_second, third = asyncio.run(exercise())

    assert first["actual_start_s"] == 0.0
    assert delayed_second["actual_start_s"] == 3.0
    assert delayed_second["start_delay_s"] == 2.0
    assert third["actual_start_s"] == 4.0
    assert third["pacing_wait_s"] == 1.0


def test_future_retry_slot_does_not_block_earlier_initial_slot(monkeypatch):
    clock = [100.0]

    async def exercise():
        retry_sleep_started = asyncio.Event()
        release_retry = asyncio.Event()

        async def controlled_sleep(delay_s):
            assert delay_s == pytest.approx(3.0)
            retry_sleep_started.set()
            await release_retry.wait()
            clock[0] = 103.0

        monkeypatch.setattr(routes.time, "perf_counter", lambda: clock[0])
        monkeypatch.setattr(routes.asyncio, "sleep", controlled_sleep)

        pacer = routes._LoadTestPacer(
            request_interval_s=1.0,
            started_at=100.0,
            initial_slots=3,
        )
        retry_task = asyncio.create_task(pacer.wait_for_slot())
        await retry_sleep_started.wait()

        clock[0] = 101.0
        initial = await asyncio.wait_for(pacer.wait_until(1.0), timeout=0.1)
        release_retry.set()
        retry = await retry_task
        return pacer, initial, retry

    pacer, initial, retry = asyncio.run(exercise())

    assert initial["actual_start_s"] == 1.0
    assert retry["scheduled_start_s"] == 3.0
    assert retry["actual_start_s"] == 3.0
    assert pacer.slots_used == 2


def test_single_request_uses_custom_payload_schema_and_paces_retries(monkeypatch):
    async def no_backoff(_delay_s):
        return None

    monkeypatch.setattr(routes.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(routes.asyncio, "sleep", no_backoff)
    workspace = FakeWorkspaceClient(
        [
            {"choices": [{"message": {"content": "not json"}}]},
            {
                "choices": [{"message": {"content": '{"custom":"ok"}'}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            },
        ]
    )

    async def exercise():
        pacer = routes._LoadTestPacer(
            request_interval_s=0.0,
            started_at=routes.time.perf_counter(),
        )
        result = await routes._run_single_load_test_request(
            workspace_client=workspace,
            endpoint_name="gpt-5-4-mini",
            endpoint_settings={},
            req_id=3,
            max_tokens=128,
            max_retries=1,
            cache_bust=True,
            prompt="Custom clinical note",
            schema_json=CUSTOM_SCHEMA,
            system_prompt="Custom system instruction",
            schema_validator=Draft202012Validator(json.loads(CUSTOM_SCHEMA)),
            attempt_pacer=pacer,
        )
        return pacer, result

    pacer, result = asyncio.run(exercise())

    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert pacer.slots_used == 2
    assert len(workspace.api_client.calls) == 2
    payload = workspace.api_client.calls[0]["body"]
    assert payload["messages"][0]["content"] == "Custom system instruction"
    assert payload["messages"][1]["content"].startswith("Custom clinical note")
    assert "benchmark req=3 nonce=" in payload["messages"][1]["content"]
    assert payload["response_format"]["json_schema"]["schema"]["required"] == [
        "custom"
    ]


def test_single_request_rejects_output_against_custom_schema(monkeypatch):
    async def no_sleep(_delay_s):
        return None

    monkeypatch.setattr(routes.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(routes.random, "uniform", lambda _a, _b: 0.0)
    invalid = {
        "choices": [{"message": {"content": '{"patient_name":"Maria"}'}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4},
    }
    workspace = FakeWorkspaceClient([invalid, invalid])

    result = asyncio.run(
        routes._run_single_load_test_request(
            workspace_client=workspace,
            endpoint_name="gpt-5-4-mini",
            endpoint_settings={},
            req_id=0,
            max_tokens=128,
            max_retries=1,
            cache_bust=False,
            prompt="Custom prompt",
            schema_json=CUSTOM_SCHEMA,
            system_prompt="",
            schema_validator=Draft202012Validator(json.loads(CUSTOM_SCHEMA)),
        )
    )

    assert result["status"] == "schema_validation_error"
    assert result["attempts"] == 2
    assert result["error"] == "The model output did not match the load-test schema."
    assert result["prompt_tokens"] == 40
    assert result["completion_tokens"] == 8
    assert result["retry_events"][0]["prompt_tokens"] == 20
    assert result["retry_events"][0]["completion_tokens"] == 4


def test_plan_endpoint_returns_authoritative_config_and_actionable_duration(monkeypatch):
    async def list_models():
        return [_model()]

    monkeypatch.setattr(routes, "list_models", list_models)
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})

    request = LoadTestRequest(
        model="gpt-5-4-mini",
        total_requests=150,
        max_workers=5,
        max_tokens=1,
        target_itpm=1,
        target_otpm=1,
        scheduling_mode="paced",
        use_documented_targets=False,
        prompt="x",
        schema_json=CUSTOM_SCHEMA,
        system_prompt="",
    )
    response = asyncio.run(routes.plan_load_test(request))

    assert response.model == request.model
    assert response.config.target_itpm == 1
    assert response.config.target_otpm == 1
    assert response.config.target_itpm_source == "custom"
    assert response.config.target_otpm_source == "custom"
    assert response.config.use_documented_targets is False
    assert response.config.estimated_input_tokens_per_request > 0
    assert response.config.planned_output_tokens_per_request == 1
    assert response.config.planned_duration_s > response.max_planned_duration_s
    assert response.runnable is False
    assert response.suggested_max_requests == 1
    assert "exceeds" in response.validation_message
    assert response.rate_limits is not None


def test_plan_reserves_endpoint_timeout_and_apps_proxy_margin(monkeypatch):
    async def list_models():
        return [_model()]

    monkeypatch.setattr(routes, "list_models", list_models)
    monkeypatch.setattr(
        routes,
        "_load_endpoint_settings",
        lambda: {"gpt-5-4-mini": {"timeout_s": 110}},
    )

    prepared = asyncio.run(
        routes._prepare_load_test(
            LoadTestRequest(
                model="gpt-5-4-mini",
                total_requests=1,
                max_workers=1,
                max_tokens=1,
            )
        )
    )

    assert prepared["max_planned_duration_s"] == pytest.approx(5.0)
    assert (
        prepared["max_planned_duration_s"]
        + 110
        + routes.LOAD_TEST_PROXY_SAFETY_MARGIN_S
    ) <= routes.DATABRICKS_APP_PROXY_TIMEOUT_S


def test_run_rejects_overlong_plan_before_workspace_access(monkeypatch):
    workspace_accessed = False

    async def list_models():
        return [_model()]

    def get_workspace_client():
        nonlocal workspace_accessed
        workspace_accessed = True
        raise AssertionError("workspace should not be accessed")

    monkeypatch.setattr(routes, "list_models", list_models)
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})
    monkeypatch.setattr(routes, "_get_workspace_client", get_workspace_client)

    response = asyncio.run(
        routes.run_load_test(
            LoadTestRequest(
                model="gpt-5-4-mini",
                total_requests=150,
                max_workers=10,
                max_tokens=1,
                target_itpm=1,
                target_otpm=1,
                scheduling_mode="paced",
            )
        )
    )
    body = json.loads(response.body)

    assert response.status_code == 422
    assert body["error_code"] == "LOAD_TEST_DURATION_EXCEEDED"
    assert "Suggested maximum requests" in body["detail"]
    assert workspace_accessed is False


def test_run_route_returns_504_and_cancels_on_wall_time_limit(monkeypatch):
    cancelled = False

    async def list_models():
        return [
            _model(
                name="custom-endpoint",
                model_type="custom",
                rate_limits=False,
            )
        ]

    async def blocked_run(**_kwargs):
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    monkeypatch.setattr(routes, "list_models", list_models)
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})
    monkeypatch.setattr(routes, "_get_workspace_client", FakeWorkspaceClient)
    monkeypatch.setattr(routes, "_run_load_test", blocked_run)
    monkeypatch.setattr(routes, "LOAD_TEST_MAX_WALL_TIME_S", 0.01)

    response = asyncio.run(
        routes.run_load_test(
            LoadTestRequest(
                model="custom-endpoint",
                total_requests=1,
                max_workers=1,
                max_tokens=1,
                use_documented_targets=False,
            )
        )
    )
    body = json.loads(response.body)

    assert response.status_code == 504
    assert body["error_code"] == "LOAD_TEST_TIMEOUT"
    assert cancelled is True


def test_prepare_load_test_uses_sample_fallbacks_and_custom_endpoint_bursts(monkeypatch):
    async def list_models():
        return [
            _model(
                name="custom-endpoint",
                model_type="custom",
                rate_limits=False,
            )
        ]

    monkeypatch.setattr(routes, "list_models", list_models)
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})

    prepared = asyncio.run(
        routes._prepare_load_test(
            LoadTestRequest(
                model="custom-endpoint",
                total_requests=2,
                max_workers=1,
                max_tokens=128,
            )
        )
    )

    assert prepared["prompt"] == routes.BENCHMARK_PROMPT
    assert prepared["schema_json"] == routes.BENCHMARK_SCHEMA_JSON
    assert prepared["system_prompt"] == routes.BENCHMARK_SYSTEM_PROMPT
    assert prepared["pacing_plan"]["pacing_mode"] == "burst"
    assert prepared["config"].target_itpm_source == "none"
    assert prepared["config"].target_otpm_source == "none"
    assert prepared["runnable"] is True


def test_run_load_test_executes_custom_endpoint_with_custom_workload(monkeypatch):
    async def list_models():
        return [
            _model(
                name="custom-endpoint",
                model_type="custom",
                rate_limits=False,
            )
        ]

    workspace = FakeWorkspaceClient(
        [
            {
                "choices": [{"message": {"content": '{"custom":"ok"}'}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 4},
            }
        ]
    )
    monkeypatch.setattr(routes, "list_models", list_models)
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: workspace)

    result = asyncio.run(
        routes.run_load_test(
            LoadTestRequest(
                model="custom-endpoint",
                total_requests=1,
                max_workers=1,
                max_tokens=64,
                use_documented_targets=False,
                prompt="A custom note",
                schema_json=CUSTOM_SCHEMA,
                system_prompt="",
            )
        )
    )

    assert result.summary.succeeded == 1
    assert result.config.pacing_mode == "burst"
    assert result.config.target_itpm is None
    assert result.config.target_otpm is None
    assert result.requests[0].scheduled_start_s == 0.0
    call = workspace.api_client.calls[0]
    assert call["path"] == "/serving-endpoints/custom-endpoint/invocations"
    assert call["body"]["messages"][0]["role"] == "user"
    assert call["body"]["messages"][0]["content"].startswith("A custom note")
    assert call["body"]["custom_inputs"]["schema"] == CUSTOM_SCHEMA


def test_run_load_test_cancels_sibling_tasks_on_unexpected_failure(monkeypatch):
    cancelled = []
    siblings_started = asyncio.Event()
    started_count = 0

    async def fake_single(*, req_id, **_kwargs):
        nonlocal started_count
        started_count += 1
        if started_count == 3:
            siblings_started.set()
        if req_id == 0:
            await siblings_started.wait()
            raise RuntimeError("unexpected")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(req_id)
            raise

    monkeypatch.setattr(routes, "_run_single_load_test_request", fake_single)
    plan = routes._build_load_test_plan(
        total_requests=3,
        estimated_input_tokens_per_request=100,
        planned_output_tokens_per_request=100,
        target_itpm=None,
        target_otpm=None,
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        asyncio.run(
            routes._run_load_test(
                workspace_client=FakeWorkspaceClient(),
                endpoint_name="custom-endpoint",
                endpoint_settings={},
                total_requests=3,
                max_workers=3,
                max_retries=0,
                max_tokens=100,
                cache_bust=False,
                use_documented_targets=False,
                target_itpm=None,
                target_otpm=None,
                target_source="none",
                target_itpm_source="none",
                target_otpm_source="none",
                estimated_input_tokens_per_request=100,
                pacing_plan=plan,
                prompt="prompt",
                schema_json=CUSTOM_SCHEMA,
                system_prompt="",
                schema_validator=Draft202012Validator(json.loads(CUSTOM_SCHEMA)),
            )
        )

    assert sorted(cancelled) == [1, 2]


def test_load_test_response_includes_plan_attainment_and_latency_percentiles():
    plan = routes._build_load_test_plan(
        total_requests=2,
        estimated_input_tokens_per_request=100,
        planned_output_tokens_per_request=20,
        target_itpm=6_000,
        target_otpm=1_200,
    )
    response = routes._build_load_test_response(
        model="custom-endpoint",
        results=[
            {
                "req_id": 0,
                "status": "ok",
                "attempts": 1,
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "elapsed_s": 1.0,
                "abs_end_s": 1.0,
                "scheduled_start_s": 0.0,
                "actual_start_s": 0.1,
                "start_delay_s": 0.1,
                "pacing_wait_s": 0.0,
                "error": None,
            },
            {
                "req_id": 1,
                "status": "ok",
                "attempts": 1,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "elapsed_s": 3.0,
                "abs_end_s": 4.0,
                "scheduled_start_s": 1.0,
                "actual_start_s": 1.25,
                "start_delay_s": 0.25,
                "pacing_wait_s": 0.8,
                "error": None,
            },
        ],
        wall_time_s=4.0,
        target_itpm=6_000,
        target_otpm=1_200,
        total_requests=2,
        max_workers=2,
        max_retries=1,
        max_tokens=20,
        cache_bust=False,
        target_source="custom",
        target_itpm_source="custom",
        target_otpm_source="custom",
        estimated_input_tokens_per_request=100,
        planned_output_tokens_per_request=20,
        pacing_plan=plan,
        pacing_slots_used=2,
    )

    assert response.config.request_interval_s == 1.0
    assert response.config.planned_itpm == 6_000.0
    assert response.config.planned_otpm == 1_200.0
    assert response.summary.achieved_itpm == 3_000.0
    assert response.summary.achieved_otpm == 450.0
    assert response.summary.itpm_target_achievement_ratio == 0.5
    assert response.summary.otpm_target_achievement_ratio == 0.375
    assert response.summary.p50_latency_s == 2.0
    assert response.summary.p95_latency_s == 2.9
    assert response.summary.average_start_delay_s == 0.175
    assert response.summary.max_start_delay_s == 0.25
    assert response.summary.actual_last_start_s == 1.25


def test_retry_invocation_updates_summary_actual_last_start(monkeypatch):
    clock = [100.0]

    async def fake_single(*, scheduled_start_s, **_kwargs):
        clock[0] = 102.0
        return {
            "req_id": 0,
            "status": "ok",
            "attempts": 2,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "elapsed_s": 2.0,
            "abs_end_s": 0.0,
            "error": None,
            "scheduled_start_s": scheduled_start_s,
            "actual_start_s": 0.0,
            "start_delay_s": 0.0,
            "pacing_wait_s": 0.0,
            "last_attempt_start_s": 2.0,
        }

    monkeypatch.setattr(routes.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(routes, "_run_single_load_test_request", fake_single)
    plan = routes._build_load_test_plan(
        total_requests=1,
        estimated_input_tokens_per_request=100,
        planned_output_tokens_per_request=20,
        target_itpm=6_000,
        target_otpm=1_200,
    )

    response = asyncio.run(
        routes._run_load_test(
            workspace_client=FakeWorkspaceClient(),
            endpoint_name="custom-endpoint",
            endpoint_settings={},
            total_requests=1,
            max_workers=1,
            max_retries=1,
            max_tokens=20,
            cache_bust=False,
            use_documented_targets=False,
            target_itpm=6_000,
            target_otpm=1_200,
            target_source="custom",
            target_itpm_source="custom",
            target_otpm_source="custom",
            estimated_input_tokens_per_request=100,
            pacing_plan=plan,
            prompt="prompt",
            schema_json=CUSTOM_SCHEMA,
            system_prompt="",
            schema_validator=Draft202012Validator(json.loads(CUSTOM_SCHEMA)),
        )
    )

    assert response.summary.total_attempts == 2
    assert response.summary.pacing_slots_used == 2
    assert response.summary.actual_last_start_s == 2.0


@pytest.fixture
def isolated_load_test_runs(monkeypatch):
    monkeypatch.setattr(routes, "_LOAD_TEST_RUNS", {})
    monkeypatch.setattr(routes, "_LOAD_TEST_PENDING_CREATIONS", 0)
    monkeypatch.setattr(routes, "_LOAD_TEST_PENDING_OWNERS", set())


def _request_for(owner: str = "clinician@example.com") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/load-test/runs",
            "headers": [(b"x-forwarded-email", owner.encode("utf-8"))],
        }
    )


def _configure_ready_async_model(monkeypatch):
    async def list_models():
        return [
            _model(
                name="custom-endpoint",
                model_type="custom",
                rate_limits=False,
            )
        ]

    monkeypatch.setattr(routes, "list_models", list_models)
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})
    monkeypatch.setattr(routes, "_get_workspace_client", FakeWorkspaceClient)


def _background_result(kwargs: dict, results: list[dict]):
    return routes._build_load_test_response(
        model=kwargs["endpoint_name"],
        results=results,
        wall_time_s=1.0,
        target_itpm=kwargs["target_itpm"],
        target_otpm=kwargs["target_otpm"],
        total_requests=kwargs["total_requests"],
        max_workers=kwargs["max_workers"],
        max_retries=kwargs["max_retries"],
        max_tokens=kwargs["max_tokens"],
        cache_bust=kwargs["cache_bust"],
        target_source=kwargs["target_source"],
        use_documented_targets=kwargs["use_documented_targets"],
        target_itpm_source=kwargs["target_itpm_source"],
        target_otpm_source=kwargs["target_otpm_source"],
        estimated_input_tokens_per_request=kwargs[
            "estimated_input_tokens_per_request"
        ],
        planned_output_tokens_per_request=kwargs["max_tokens"],
        pacing_plan=kwargs["pacing_plan"],
    )


def test_load_test_request_defaults_to_burst_with_targets_as_references(monkeypatch):
    _configure_ready_async_model(monkeypatch)
    request = LoadTestRequest(
        model="custom-endpoint",
        total_requests=150,
        max_workers=10,
        target_itpm=20_000,
        target_otpm=20_000,
        use_documented_targets=False,
    )

    prepared = asyncio.run(routes._prepare_load_test(request))

    assert request.scheduling_mode == "burst"
    assert prepared["config"].pacing_mode == "burst"
    assert prepared["config"].target_itpm == 20_000
    assert prepared["config"].target_otpm == 20_000
    assert prepared["config"].planned_duration_s == 0.0
    assert prepared["runnable"] is True


def test_create_run_route_returns_http_202(monkeypatch, isolated_load_test_runs):
    from fastapi.testclient import TestClient
    from main import app

    _configure_ready_async_model(monkeypatch)

    async def fake_run(**kwargs):
        row = {
            "req_id": 0,
            "status": "ok",
            "attempts": 1,
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "elapsed_s": 0.1,
            "abs_end_s": 0.1,
            "error": None,
        }
        kwargs["progress_callback"](row)
        return _background_result(kwargs, [row])

    monkeypatch.setattr(routes, "_run_load_test", fake_run)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/load-test/runs",
            headers={"x-forwarded-email": "clinician@example.com"},
            json={
                "model": "custom-endpoint",
                "total_requests": 1,
                "max_workers": 1,
                "max_tokens": 64,
                "use_documented_targets": False,
                "scheduling_mode": "burst",
            },
        )

    assert response.status_code == 202
    assert response.json()["status"] in {"queued", "running", "completed"}
    assert response.json()["model"] == "custom-endpoint"
    assert response.json()["config"]["pacing_mode"] == "burst"


def test_background_run_reports_progress_result_and_active_recovery(
    monkeypatch,
    isolated_load_test_runs,
):
    _configure_ready_async_model(monkeypatch)

    async def scenario():
        progress_reported = asyncio.Event()
        release = asyncio.Event()
        first_row = {
            "req_id": 0,
            "status": "ok",
            "attempts": 1,
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "elapsed_s": 0.1,
            "abs_end_s": 0.1,
            "error": None,
        }
        second_row = {
            "req_id": 1,
            "status": "429_exhausted",
            "attempts": 2,
            "prompt_tokens": None,
            "completion_tokens": None,
            "elapsed_s": 0.2,
            "abs_end_s": 0.2,
            "error": "The endpoint rate limit was exceeded after all retries.",
        }

        async def fake_run(**kwargs):
            kwargs["progress_callback"](first_row)
            progress_reported.set()
            await release.wait()
            return _background_result(kwargs, [first_row, second_row])

        monkeypatch.setattr(routes, "_run_load_test", fake_run)
        owner_request = _request_for()
        created = await routes.create_load_test_run(
            owner_request,
            LoadTestRequest(
                model="custom-endpoint",
                total_requests=2,
                max_workers=1,
                max_tokens=64,
                target_otpm=20_000,
                use_documented_targets=False,
            ),
        )
        assert created.status == "queued"

        await progress_reported.wait()
        running = await routes.get_load_test_run(created.run_id, owner_request)
        assert running.status == "running"
        assert running.progress.completed_requests == 1
        assert running.progress.succeeded == 1
        assert running.progress.total_attempts == 1
        assert running.partial_result is not None
        assert len(running.partial_result.requests) == 1
        assert len(running.partial_result.series) == 1
        assert running.partial_result.summary.total_completion_tokens == 4
        assert running.partial_result.series[0].cumulative_completion_tokens == 4
        assert running.partial_result.summary.target_otpm == 20_000

        active = await routes.list_active_load_test_runs(owner_request)
        assert [run.run_id for run in active] == [created.run_id]
        duplicate = await routes.create_load_test_run(
            owner_request,
            LoadTestRequest(
                model="custom-endpoint",
                total_requests=1,
                max_workers=1,
                max_tokens=64,
                use_documented_targets=False,
            ),
        )
        assert isinstance(duplicate, JSONResponse)
        assert duplicate.status_code == 409
        assert created.run_id in json.loads(duplicate.body)["detail"]
        assert await routes.list_active_load_test_runs(
            _request_for("other@example.com")
        ) == []

        task = routes._LOAD_TEST_RUNS[created.run_id].task
        release.set()
        await task
        await asyncio.sleep(0)
        completed = await routes.get_load_test_run(created.run_id, owner_request)
        assert completed.status == "completed"
        assert completed.result is not None
        assert completed.partial_result is None
        assert completed.progress.completed_requests == 2
        assert completed.progress.succeeded == 1
        assert completed.progress.failed == 1
        assert completed.progress.total_attempts == 3
        assert await routes.list_active_load_test_runs(owner_request) == []

    asyncio.run(scenario())


def test_background_run_limit_is_atomic_during_preparation(
    monkeypatch,
    isolated_load_test_runs,
):
    _configure_ready_async_model(monkeypatch)
    monkeypatch.setattr(routes, "LOAD_TEST_MAX_ACTIVE_RUNS", 1)

    async def scenario():
        request = LoadTestRequest(
            model="custom-endpoint",
            total_requests=1,
            max_workers=1,
            max_tokens=64,
            use_documented_targets=False,
        )
        original_prepare = routes._prepare_load_test
        prepared = await original_prepare(
            request,
            max_planned_duration_s=routes.LOAD_TEST_BACKGROUND_MAX_RUNTIME_S,
            max_wall_time_s=routes.LOAD_TEST_BACKGROUND_MAX_RUNTIME_S,
        )
        prepare_started = asyncio.Event()
        release_prepare = asyncio.Event()

        async def blocked_prepare(*_args, **_kwargs):
            prepare_started.set()
            await release_prepare.wait()
            return prepared

        async def blocked_run(**_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(routes, "_prepare_load_test", blocked_prepare)
        monkeypatch.setattr(routes, "_run_load_test", blocked_run)
        owner_request = _request_for()
        first_creation = asyncio.create_task(
            routes.create_load_test_run(owner_request, request)
        )
        await prepare_started.wait()

        duplicate = await routes.create_load_test_run(owner_request, request)
        assert isinstance(duplicate, JSONResponse)
        assert duplicate.status_code == 409

        rejected = await routes.create_load_test_run(
            _request_for("other@example.com"), request
        )
        assert isinstance(rejected, JSONResponse)
        assert rejected.status_code == 429

        release_prepare.set()
        created = await first_creation
        await routes.cancel_load_test_run(created.run_id, owner_request)
        task = routes._LOAD_TEST_RUNS[created.run_id].task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_background_run_cancellation_and_timeout(
    monkeypatch,
    isolated_load_test_runs,
):
    _configure_ready_async_model(monkeypatch)

    async def cancellation_scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_run(**_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(routes, "_run_load_test", blocked_run)
        owner_request = _request_for()
        created = await routes.create_load_test_run(
            owner_request,
            LoadTestRequest(
                model="custom-endpoint",
                total_requests=1,
                max_workers=1,
                max_tokens=64,
                use_documented_targets=False,
            ),
        )
        await started.wait()
        task = routes._LOAD_TEST_RUNS[created.run_id].task
        response = await routes.cancel_load_test_run(created.run_id, owner_request)
        assert response.status == "cancelled"
        assert response.completed_at is not None
        assert response.expires_at is not None
        await asyncio.gather(task, return_exceptions=True)
        assert cancelled.is_set()

    asyncio.run(cancellation_scenario())

    routes._LOAD_TEST_RUNS.clear()
    monkeypatch.setattr(routes, "LOAD_TEST_BACKGROUND_MAX_RUNTIME_S", 0.01)

    async def timeout_scenario():
        async def blocked_run(**_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(routes, "_run_load_test", blocked_run)
        owner_request = _request_for()
        created = await routes.create_load_test_run(
            owner_request,
            LoadTestRequest(
                model="custom-endpoint",
                total_requests=1,
                max_workers=1,
                max_tokens=64,
                use_documented_targets=False,
            ),
        )
        task = routes._LOAD_TEST_RUNS[created.run_id].task
        await task
        failed = await routes.get_load_test_run(created.run_id, owner_request)
        assert failed.status == "failed"
        assert failed.error is not None
        assert failed.error.error_code == "LOAD_TEST_TIMEOUT"
        assert "15-minute" in failed.error.message

    asyncio.run(timeout_scenario())


def test_run_ttl_cleanup_and_unknown_run_responses(
    monkeypatch,
    isolated_load_test_runs,
):
    _configure_ready_async_model(monkeypatch)

    async def scenario():
        async def fake_run(**kwargs):
            return _background_result(kwargs, [])

        monkeypatch.setattr(routes, "_run_load_test", fake_run)
        owner_request = _request_for()
        created = await routes.create_load_test_run(
            owner_request,
            LoadTestRequest(
                model="custom-endpoint",
                total_requests=1,
                max_workers=1,
                max_tokens=64,
                use_documented_targets=False,
            ),
        )
        task = routes._LOAD_TEST_RUNS[created.run_id].task
        await task
        await asyncio.sleep(0)
        routes._LOAD_TEST_RUNS[created.run_id].response.expires_at = (
            routes._utc_now() - timedelta(seconds=1)
        )
        routes._cleanup_load_test_runs()
        assert created.run_id not in routes._LOAD_TEST_RUNS

        missing_get = await routes.get_load_test_run("missing", owner_request)
        missing_delete = await routes.cancel_load_test_run(
            "missing", owner_request
        )
        assert missing_get.status_code == 404
        assert missing_delete.status_code == 404

    asyncio.run(scenario())


def test_async_plan_uses_background_budget_while_sync_keeps_proxy_budget(
    monkeypatch,
):
    _configure_ready_async_model(monkeypatch)
    request = LoadTestRequest(
        model="custom-endpoint",
        total_requests=2,
        max_workers=1,
        max_tokens=64,
        target_otpm=1,
        use_documented_targets=False,
        scheduling_mode="paced",
    )

    sync_prepared = asyncio.run(routes._prepare_load_test(request))
    async_plan = asyncio.run(routes.plan_load_test(request))

    assert sync_prepared["max_planned_duration_s"] <= 90.0
    assert async_plan.max_planned_duration_s > sync_prepared[
        "max_planned_duration_s"
    ]
