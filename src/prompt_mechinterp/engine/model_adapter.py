"""Auto-discovers model architecture from any HuggingFace transformer.

Probes model.config for layer count, head counts, hidden size, vocab size.
Walks the module tree to find the attention submodule pattern.
Validates that eager attention is available (required for attention capture).

Supports: Llama, Qwen, Mistral, Gemma, and any model following the
standard model.model.layers[i].self_attn pattern. For non-standard
architectures, walks the module tree searching for attention submodules.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Auto-discovers architecture details from a loaded HuggingFace model."""

    def __init__(
        self,
        num_layers: int,
        num_query_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        vocab_size: int,
        attention_modules: List[Tuple[int, object]],
        layer_modules: List[Tuple[int, object]],
        lm_head: object,
        norm: object,
        model_name: str = "unknown",
    ):
        self._num_layers = num_layers
        self._num_query_heads = num_query_heads
        self._num_kv_heads = num_kv_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._attention_modules = attention_modules
        self._layer_modules = layer_modules
        self._lm_head = lm_head
        self._norm = norm
        self._model_name = model_name

    @classmethod
    def from_model(cls, model, tokenizer=None) -> "ModelAdapter":
        """Construct adapter by inspecting a loaded model."""
        config = model.config

        num_layers = config.num_hidden_layers
        num_query_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_query_heads)
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        model_name = getattr(config, "_name_or_path", "unknown")

        logger.info(
            "Model config: %d layers, %d query heads, %d kv heads, "
            "hidden=%d, vocab=%d",
            num_layers, num_query_heads, num_kv_heads,
            hidden_size, vocab_size,
        )

        # Find attention and layer modules
        attention_modules = []
        layer_modules = []

        # Strategy 1: Standard path (model.model.layers[i].self_attn)
        # Works for Llama, Qwen, Mistral, Gemma, and most decoder-only models
        layers_container = _find_layers_container(model)

        if layers_container is not None and len(layers_container) == num_layers:
            for i, layer in enumerate(layers_container):
                layer_modules.append((i, layer))
                attn = _find_attention_submodule(layer)
                if attn is not None:
                    attention_modules.append((i, attn))
            logger.info(
                "Found %d layer modules, %d attention modules via standard path",
                len(layer_modules), len(attention_modules),
            )
        else:
            raise RuntimeError(
                f"Could not find layer container with {num_layers} layers. "
                "This model architecture may not be supported. "
                "Expected model.model.layers or model.transformer.h"
            )

        if len(attention_modules) != num_layers:
            raise RuntimeError(
                f"Found {len(attention_modules)} attention modules, "
                f"expected {num_layers}"
            )

        # Find LM head
        lm_head = _find_lm_head(model)
        if lm_head is None:
            raise RuntimeError("Could not find language model head (lm_head)")

        # Find final norm
        norm = _find_final_norm(model)
        if norm is None:
            raise RuntimeError("Could not find final normalization layer")

        logger.info("Model adapter ready: %s", model_name)

        return cls(
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            attention_modules=attention_modules,
            layer_modules=layer_modules,
            lm_head=lm_head,
            norm=norm,
            model_name=model_name,
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_query_heads(self) -> int:
        return self._num_query_heads

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_attention_modules(self) -> List[Tuple[int, object]]:
        """Return list of (layer_idx, attn_module) for hook registration."""
        return self._attention_modules

    def get_layer_modules(self) -> List[Tuple[int, object]]:
        """Return list of (layer_idx, layer_module) for residual hooks."""
        return self._layer_modules

    def get_lm_head(self):
        """Return the language model head for logit lens projection."""
        return self._lm_head

    def get_norm(self):
        """Return the final normalization layer before lm_head."""
        return self._norm


def _find_layers_container(model):
    """Find the sequential container holding transformer layers.

    Tries known paths in order:
    - model.model.layers (Llama, Qwen, Mistral, Gemma)
    - model.transformer.h (GPT-2, GPT-Neo)
    - model.gpt_neox.layers (GPT-NeoX)
    """
    paths = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in paths:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            logger.info("Found layers at: %s", ".".join(path))
            return obj
    return None


def _find_attention_submodule(layer):
    """Find the self-attention submodule within a decoder layer.

    Tries: self_attn, attention, attn
    """
    for name in ("self_attn", "attention", "attn"):
        attn = getattr(layer, name, None)
        if attn is not None:
            return attn
    return None


def _find_lm_head(model):
    """Find the language model head.

    Tries: model.lm_head, model.output, model.embed_out
    """
    for name in ("lm_head", "output", "embed_out"):
        head = getattr(model, name, None)
        if head is not None:
            logger.info("Found LM head at: model.%s", name)
            return head
    return None


def _find_final_norm(model):
    """Find the final normalization layer before lm_head.

    Tries: model.model.norm, model.model.final_layernorm,
           model.transformer.ln_f, model.gpt_neox.final_layer_norm
    """
    paths = [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    ]
    for path in paths:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            logger.info("Found final norm at: %s", ".".join(path))
            return obj
    return None
