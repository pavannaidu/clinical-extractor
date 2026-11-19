import mlflow
from mlflow.types.llm import (
    ChatCompletionResponse,
    ChatCompletionRequest,
    ChatChoice,
    ChatMessage,
    ChatParams,
    TokenUsageStats,
)
from typing import Optional


class MedGemmaChatModel(mlflow.pyfunc.ChatModel):
    def load_context(self, context):
        import torch, xgrammar as xgr
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_path = context.artifacts["model-weights"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        print(f"[MedGemma] GPU={torch.cuda.get_device_name()}, dtype=bf16, attn=sdpa")

        vocab_size = getattr(
            self.model.config,
            "vocab_size",
            getattr(getattr(self.model.config, "text_config", self.model.config), "vocab_size", None)
        )
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=vocab_size
        )
        self.compiler = xgr.GrammarCompiler(tokenizer_info)
        self._grammar_cache = {}

        # Warm up CUDA kernels to avoid JIT compilation on first requests
        dummy = self.tokenizer("warmup", return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model.generate(**dummy, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.empty_cache()
        print("[MedGemma] Warmup complete")

    def _generate(self, prompt_text: str, schema_str: Optional[str], max_new_tokens: int = 256, temperature: float = 0.7):
        """Generate a single completion. Returns (text, prompt_tokens, completion_tokens)."""
        import torch, xgrammar as xgr

        if schema_str:
            if schema_str not in self._grammar_cache:
                # Unbounded whitespace can consume the token budget before
                # the model finishes an otherwise schema-constrained object.
                self._grammar_cache[schema_str] = self.compiler.compile_json_schema(
                    schema_str, any_whitespace=False
                )
            compiled = self._grammar_cache[schema_str]
        else:
            compiled = None

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        logits_processors = [xgr.contrib.hf.LogitsProcessor(compiled)] if compiled else None
        sampling = {"do_sample": temperature > 0}
        if temperature > 0:
            sampling["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **sampling,
                logits_processor=logits_processors,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[:, prompt_tokens:]
        completion_tokens = new_tokens.shape[1]
        text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        torch.cuda.empty_cache()
        return text, prompt_tokens, completion_tokens

    def predict(self, context, messages: list[ChatMessage], params: ChatParams) -> ChatCompletionResponse:
        import json

        # Extract schema from custom_inputs
        custom_inputs = params.custom_inputs or {} if params else {}
        schema_val = custom_inputs.get("schema")
        schema_str = json.dumps(json.loads(schema_val)) if schema_val else None

        max_tokens = params.max_tokens if params and params.max_tokens else 256
        temperature = params.temperature if params and params.temperature is not None else 0.7

        # Build chat messages for the template
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Apply chat template with full conversation history
        prompt_text = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        text, prompt_tokens, completion_tokens = self._generate(
            prompt_text, schema_str, max_new_tokens=max_tokens, temperature=temperature
        )

        return ChatCompletionResponse(
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=text))],
            usage=TokenUsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


mlflow.models.set_model(MedGemmaChatModel())
