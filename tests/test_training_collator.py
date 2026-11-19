"""Exercise notebook training code without downloading weights or starting GPUs."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_notebook_definition(name, namespace=None):
    notebook = Path(__file__).resolve().parents[1] / "src/models/medgemma/finetune.py"
    tree = ast.parse(notebook.read_text())
    definition = next(
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )
    # Other notebook top-level cells install dependencies and train models.
    namespace = namespace if namespace is not None else {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_gemma_labels_follow_input_padding(padding_side):
    collator = load_notebook_definition(
        "Gemma3TextCollator",
        {"torch": SimpleNamespace(tensor=np.asarray, zeros_like=np.zeros_like)},
    )

    class Tokenizer:
        def pad(self, features, **kwargs):
            rows = features["input_ids"]
            width = max(map(len, rows))
            padded = []
            for row in rows:
                padding = [0] * (width - len(row))
                padded.append(padding + row if self.padding_side == "left" else row + padding)
            ids = np.asarray(padded)
            return {"input_ids": ids, "attention_mask": (ids != 0).astype(int)}

    tokenizer = Tokenizer()
    tokenizer.padding_side = padding_side
    batch = collator(tokenizer)([
        {"input_ids": [11, 12, 13], "labels": [11, 12, 13]},
        {"input_ids": [21], "labels": [21]},
    ])

    attended = batch["attention_mask"].astype(bool)
    np.testing.assert_array_equal(batch["labels"][attended], batch["input_ids"][attended])
    assert np.all(batch["labels"][~attended] == -100)
    assert np.all(batch["token_type_ids"] == 0)


@pytest.mark.parametrize("prefix", ["", "language_model.", "model.language_model."])
def test_merged_export_copies_tied_head_for_each_model_prefix(prefix):
    prepare_state = load_notebook_definition("merged_state_for_save")

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def clone(self):
            return Tensor(self.values.copy())

    embeddings = Tensor([[1, 2], [3, 4]])
    weights = {prefix + "lm_head.weight": embeddings, prefix + "embed_tokens.weight": embeddings}
    model = SimpleNamespace(state_dict=lambda: weights.copy())
    result = prepare_state(model)

    np.testing.assert_array_equal(result[prefix + "lm_head.weight"].values, embeddings.values)
    assert not np.shares_memory(result[prefix + "lm_head.weight"].values, embeddings.values)
    assert weights[prefix + "lm_head.weight"] is embeddings
    assert result[prefix + "embed_tokens.weight"] is embeddings
