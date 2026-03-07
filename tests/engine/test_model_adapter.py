"""Tests for engine/model_adapter.py — architecture discovery via getattr probing.

Uses types.SimpleNamespace to simulate PyTorch model structure, which is
exactly how PyTorch models expose submodules via attribute access.
"""

from types import SimpleNamespace

import pytest

from prompt_mechinterp.engine.model_adapter import (
    ModelAdapter,
    _find_attention_submodule,
    _find_final_norm,
    _find_layers_container,
    _find_lm_head,
)


def _make_layer(attn_attr="self_attn"):
    """Create a mock decoder layer with an attention submodule."""
    attn = SimpleNamespace(name="attention_module")
    layer = SimpleNamespace(**{attn_attr: attn})
    return layer


def _make_llama_model(num_layers=4):
    """Build a complete Llama-style mock model."""
    layers = [_make_layer("self_attn") for _ in range(num_layers)]
    norm = SimpleNamespace(name="final_norm")
    inner = SimpleNamespace(layers=layers, norm=norm)
    lm_head = SimpleNamespace(name="lm_head")
    config = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        vocab_size=32000,
        _name_or_path="test/llama-mock",
    )
    return SimpleNamespace(model=inner, lm_head=lm_head, config=config)


# ===========================================================================
# _find_layers_container
# ===========================================================================

class TestFindLayersContainer:
    def test_llama_style(self):
        layers = [SimpleNamespace() for _ in range(32)]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        result = _find_layers_container(model)
        assert result is layers

    def test_gpt2_style(self):
        h = [SimpleNamespace() for _ in range(12)]
        model = SimpleNamespace(transformer=SimpleNamespace(h=h))
        result = _find_layers_container(model)
        assert result is h

    def test_gpt_neox_style(self):
        layers = [SimpleNamespace() for _ in range(24)]
        model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))
        result = _find_layers_container(model)
        assert result is layers

    def test_no_matching_path(self):
        model = SimpleNamespace(encoder=SimpleNamespace(blocks=[]))
        result = _find_layers_container(model)
        assert result is None

    def test_partial_path_returns_none(self):
        """model.model exists but has no .layers attribute."""
        model = SimpleNamespace(model=SimpleNamespace())
        result = _find_layers_container(model)
        assert result is None


# ===========================================================================
# _find_attention_submodule
# ===========================================================================

class TestFindAttentionSubmodule:
    def test_self_attn(self):
        attn = SimpleNamespace()
        layer = SimpleNamespace(self_attn=attn)
        assert _find_attention_submodule(layer) is attn

    def test_attention(self):
        attn = SimpleNamespace()
        layer = SimpleNamespace(attention=attn)
        assert _find_attention_submodule(layer) is attn

    def test_attn(self):
        attn = SimpleNamespace()
        layer = SimpleNamespace(attn=attn)
        assert _find_attention_submodule(layer) is attn

    def test_none_when_no_match(self):
        layer = SimpleNamespace(mlp=SimpleNamespace())
        assert _find_attention_submodule(layer) is None

    def test_priority_order(self):
        """self_attn should be found first even if attention also exists."""
        self_attn = SimpleNamespace(name="self_attn")
        attention = SimpleNamespace(name="attention")
        layer = SimpleNamespace(self_attn=self_attn, attention=attention)
        assert _find_attention_submodule(layer) is self_attn


# ===========================================================================
# _find_lm_head
# ===========================================================================

class TestFindLmHead:
    def test_lm_head(self):
        head = SimpleNamespace()
        model = SimpleNamespace(lm_head=head)
        assert _find_lm_head(model) is head

    def test_output(self):
        head = SimpleNamespace()
        model = SimpleNamespace(output=head)
        assert _find_lm_head(model) is head

    def test_embed_out(self):
        head = SimpleNamespace()
        model = SimpleNamespace(embed_out=head)
        assert _find_lm_head(model) is head

    def test_none_when_missing(self):
        model = SimpleNamespace()
        assert _find_lm_head(model) is None


# ===========================================================================
# _find_final_norm
# ===========================================================================

class TestFindFinalNorm:
    def test_llama_norm(self):
        norm = SimpleNamespace()
        model = SimpleNamespace(model=SimpleNamespace(norm=norm))
        assert _find_final_norm(model) is norm

    def test_gpt2_ln_f(self):
        norm = SimpleNamespace()
        model = SimpleNamespace(transformer=SimpleNamespace(ln_f=norm))
        assert _find_final_norm(model) is norm

    def test_gpt_neox_final_layer_norm(self):
        norm = SimpleNamespace()
        model = SimpleNamespace(gpt_neox=SimpleNamespace(final_layer_norm=norm))
        assert _find_final_norm(model) is norm

    def test_final_layernorm(self):
        norm = SimpleNamespace()
        model = SimpleNamespace(model=SimpleNamespace(final_layernorm=norm))
        assert _find_final_norm(model) is norm

    def test_none_when_missing(self):
        model = SimpleNamespace()
        assert _find_final_norm(model) is None


# ===========================================================================
# ModelAdapter.from_model
# ===========================================================================

class TestModelAdapterFromModel:
    def test_full_llama_model(self):
        model = _make_llama_model(num_layers=4)
        adapter = ModelAdapter.from_model(model)

        assert adapter.num_layers == 4
        assert adapter.num_query_heads == 32
        assert adapter.num_kv_heads == 8
        assert adapter.hidden_size == 4096
        assert adapter.vocab_size == 32000
        assert "llama-mock" in adapter.model_name
        assert len(adapter.get_attention_modules()) == 4
        assert len(adapter.get_layer_modules()) == 4
        assert adapter.get_lm_head() is not None
        assert adapter.get_norm() is not None

    def test_missing_attention_raises(self):
        """If attention submodule can't be found in layers, should raise."""
        layers = [SimpleNamespace(mlp=SimpleNamespace()) for _ in range(4)]
        norm = SimpleNamespace()
        inner = SimpleNamespace(layers=layers, norm=norm)
        lm_head = SimpleNamespace()
        config = SimpleNamespace(
            num_hidden_layers=4,
            num_attention_heads=32,
            hidden_size=4096,
            vocab_size=32000,
        )
        model = SimpleNamespace(model=inner, lm_head=lm_head, config=config)

        with pytest.raises(RuntimeError, match="attention"):
            ModelAdapter.from_model(model)

    def test_missing_lm_head_raises(self):
        layers = [_make_layer() for _ in range(4)]
        norm = SimpleNamespace()
        inner = SimpleNamespace(layers=layers, norm=norm)
        config = SimpleNamespace(
            num_hidden_layers=4,
            num_attention_heads=32,
            hidden_size=4096,
            vocab_size=32000,
        )
        model = SimpleNamespace(model=inner, config=config)

        with pytest.raises(RuntimeError, match="lm_head"):
            ModelAdapter.from_model(model)

    def test_missing_norm_raises(self):
        layers = [_make_layer() for _ in range(4)]
        inner = SimpleNamespace(layers=layers)
        lm_head = SimpleNamespace()
        config = SimpleNamespace(
            num_hidden_layers=4,
            num_attention_heads=32,
            hidden_size=4096,
            vocab_size=32000,
        )
        model = SimpleNamespace(model=inner, lm_head=lm_head, config=config)

        with pytest.raises(RuntimeError, match="norm"):
            ModelAdapter.from_model(model)

    def test_layer_count_mismatch_raises(self):
        """Container with wrong number of layers should raise."""
        layers = [_make_layer() for _ in range(3)]  # Config says 4
        norm = SimpleNamespace()
        inner = SimpleNamespace(layers=layers, norm=norm)
        lm_head = SimpleNamespace()
        config = SimpleNamespace(
            num_hidden_layers=4,
            num_attention_heads=32,
            hidden_size=4096,
            vocab_size=32000,
        )
        model = SimpleNamespace(model=inner, lm_head=lm_head, config=config)

        with pytest.raises(RuntimeError, match="layer container"):
            ModelAdapter.from_model(model)

    def test_kv_heads_defaults_to_query_heads(self):
        """Models without GQA should have kv_heads == query_heads."""
        model = _make_llama_model(num_layers=4)
        del model.config.num_key_value_heads
        adapter = ModelAdapter.from_model(model)
        assert adapter.num_kv_heads == adapter.num_query_heads
