"""Keep Spark driver inference independent of torchrun environment detection."""

import ast
import builtins
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow.pyfunc.loaders.chat_model import _ChatModelPyfuncWrapper


@pytest.mark.parametrize("world_size, fail", [(None, False), ("1", False), ("1", True)])
def test_driver_load_ignores_world_size_and_restores_it(monkeypatch, world_size, fail):
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    generation_config = SimpleNamespace(disable_compile=False)
    chat_model = SimpleNamespace(model=SimpleNamespace(generation_config=generation_config))
    sentinel = PyFuncModel(Model(), _ChatModelPyfuncWrapper(chat_model, None, None))

    def load_model(uri):
        assert uri == "models:/test_model@challenger"
        assert "WORLD_SIZE" not in os.environ
        if fail:
            raise RuntimeError("model load failed")
        return sentinel

    notebook = Path(__file__).resolve().parents[1] / "src/models/medgemma/batch/02_batch_inference_udf.py"
    # Databricks %pip cells are notebook syntax, so parse only the loader cell.
    cell = next(cell for cell in notebook.read_text().split("# COMMAND ----------")
                if "def load_driver_model(" in cell)
    definition = next(node for node in ast.parse(cell).body if isinstance(node, ast.FunctionDef))
    namespace = {"mlflow": SimpleNamespace(pyfunc=SimpleNamespace(load_model=load_model))}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(notebook), "exec"), namespace)

    original_import = builtins.__import__

    def runtime_import(name, *args, **kwargs):
        if name == "torch":
            # DBR sets this on first import, even when the notebook started
            # without any torchrun environment variables.
            os.environ.setdefault("WORLD_SIZE", "1")
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", runtime_import)

    if fail:
        with pytest.raises(RuntimeError, match="model load failed"):
            namespace["load_driver_model"]("models:/test_model@challenger")
    else:
        assert namespace["load_driver_model"]("models:/test_model@challenger") is sentinel
        assert generation_config.disable_compile is True
    assert os.environ.get("WORLD_SIZE") == (world_size or "1")
