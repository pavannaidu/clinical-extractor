"""Check generation controls without downloading a model or requiring a GPU."""

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

from mlflow.types.llm import ChatChoice, ChatCompletionResponse, ChatMessage, ChatParams, TokenUsageStats
import pytest


def model_class():
    path = Path(__file__).resolve().parents[1] / "src/models/medgemma/xgrammar/model.py"
    definition = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    definition.bases = []
    namespace = dict(Optional=Optional, ChatMessage=ChatMessage, ChatParams=ChatParams,
                     ChatCompletionResponse=ChatCompletionResponse, ChatChoice=ChatChoice,
                     TokenUsageStats=TokenUsageStats)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[definition.name]


@pytest.mark.parametrize("temperature, expected", [(None, 0.7), (0.0, 0.0), (0.5, 0.5)])
def test_predict_preserves_explicit_zero_temperature(temperature, expected):
    model = model_class()()
    model.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")
    seen = {}

    def generate(prompt, schema, **kwargs):
        seen.update(kwargs)
        return "{}", 1, 1

    model._generate = generate
    params = ChatParams(temperature=temperature) if temperature is not None else None
    response = model.predict(None, [ChatMessage(role="user", content="test")], params)
    assert seen["temperature"] == expected
    assert response.choices[0].message.content == "{}"


@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_generation_bounds_whitespace_caches_grammar_and_controls_sampling(monkeypatch, temperature):
    class Tokens(dict):
        def to(self, device):
            return self

    class Output:
        shape = (1, 2)

        def __getitem__(self, key):
            return self

    class Tokenizer:
        eos_token_id = 1

        def __call__(self, *args, **kwargs):
            return Tokens(input_ids=SimpleNamespace(shape=(1, 3)))

        def decode(self, *args, **kwargs):
            return '{"heart_rate": 72}'

    compiled = object()
    calls = []
    generations = []

    def compile_schema(schema, **kwargs):
        calls.append((schema, kwargs))
        return compiled

    def generate(**kwargs):
        generations.append(kwargs)
        return Output()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext,
                        cuda=SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setitem(sys.modules, "xgrammar", SimpleNamespace(contrib=SimpleNamespace(
        hf=SimpleNamespace(LogitsProcessor=lambda grammar: grammar))))
    model = model_class()()
    model.tokenizer = Tokenizer()
    model.model = SimpleNamespace(device="cuda:0", generate=generate)
    model.compiler = SimpleNamespace(compile_json_schema=compile_schema)
    model._grammar_cache = {}
    schema = json.dumps({"type": "object", "properties": {"heart_rate": {"type": "integer"}}})

    for _ in range(2):
        text, prompt_tokens, completion_tokens = model._generate("prompt", schema, temperature=temperature)
        assert json.loads(text) == {"heart_rate": 72}
        assert (prompt_tokens, completion_tokens) == (3, 2)

    assert calls == [(schema, {"any_whitespace": False})]
    for kwargs in generations:
        assert kwargs["logits_processor"] == [compiled]
        assert kwargs["do_sample"] == (temperature > 0)
        if temperature == 0:
            assert "temperature" not in kwargs
        else:
            assert kwargs["temperature"] == temperature
