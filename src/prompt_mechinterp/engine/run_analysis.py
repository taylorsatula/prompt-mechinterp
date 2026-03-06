#!/usr/bin/env python3
"""Mechanistic interpretability analysis for any HuggingFace transformer model.

Self-contained — no local package imports. Designed to be scp'd to a remote
GPU box and executed standalone. Loads test_cases.json, runs forward pass
with attention extraction + logit lens, writes per-case result JSON.

Auto-detects model architecture (layer count, head count, attention module
paths, LM head, final norm) from model.config and module tree inspection.

Usage:
    python3 run_analysis.py \\
        --input test_cases.json \\
        --output ./results/ \\
        --model-path /workspace/models/Qwen3-32B

    python3 run_analysis.py \\
        --input test_cases.json \\
        --output ./results/ \\
        --model-path /workspace/models/Llama-3-8B \\
        --tracked-tokens "<" "folder_a" \\
        --top-k 50
"""

import argparse
import bisect
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# MODEL AUTO-DISCOVERY (inlined from model_adapter.py for self-containment)
# ============================================================================

def _discover_model_config(model) -> dict:
    """Auto-detect model architecture from config and module tree.

    Returns dict with: num_layers, num_query_heads, num_kv_heads,
    hidden_size, vocab_size, attention_modules, layer_modules,
    lm_head, norm, model_name.
    """
    config = model.config

    num_layers = config.num_hidden_layers
    num_query_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_query_heads)
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    model_name = getattr(config, "_name_or_path", "unknown")

    print(f"  Model config: {num_layers} layers, {num_query_heads} query heads, "
          f"{num_kv_heads} kv heads, hidden={hidden_size}, vocab={vocab_size}")

    # Find layers container
    layers_container = None
    layers_path = None
    for path_parts in [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]:
        obj = model
        for attr in path_parts:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            layers_container = obj
            layers_path = ".".join(path_parts)
            break

    if layers_container is None or len(layers_container) != num_layers:
        raise RuntimeError(
            f"Could not find layer container with {num_layers} layers"
        )

    print(f"  Layers found at: model.{layers_path} ({len(layers_container)} layers)")

    # Find attention submodules
    attention_modules = []
    layer_modules = []
    for i, layer in enumerate(layers_container):
        layer_modules.append((i, layer))
        attn = None
        for name in ("self_attn", "attention", "attn"):
            attn = getattr(layer, name, None)
            if attn is not None:
                break
        if attn is not None:
            attention_modules.append((i, attn))

    if len(attention_modules) != num_layers:
        raise RuntimeError(
            f"Found {len(attention_modules)} attention modules, expected {num_layers}"
        )

    # Find LM head
    lm_head = None
    for name in ("lm_head", "output", "embed_out"):
        lm_head = getattr(model, name, None)
        if lm_head is not None:
            print(f"  LM head at: model.{name}")
            break
    if lm_head is None:
        raise RuntimeError("Could not find language model head")

    # Find final norm
    norm = None
    for path_parts in [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    ]:
        obj = model
        for attr in path_parts:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            norm = obj
            print(f"  Final norm at: model.{'.'.join(path_parts)}")
            break
    if norm is None:
        raise RuntimeError("Could not find final normalization layer")

    return {
        "num_layers": num_layers,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "attention_modules": attention_modules,
        "layer_modules": layer_modules,
        "lm_head": lm_head,
        "norm": norm,
        "model_name": model_name,
    }


# ============================================================================
# TOKENIZATION & REGION MAPPING
# ============================================================================

def build_chat_tokens(
    tokenizer,
    system_prompt: str,
    user_message: str,
    response: str,
) -> Tuple[List[int], Dict[str, Tuple[int, int]]]:
    """Build token sequence using the model's chat template with piece boundaries.

    Handles models that don't support system role (e.g. Gemma) by merging
    system content into the user message. Uses char-level search in decoded
    text + cumulative decode mapping for robust piece boundary detection
    across all chat template formats (ChatML, Llama, Mistral, Gemma, etc.).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]

    try:
        reference_ids = _apply_template(tokenizer, messages)
    except Exception as e:
        if "system" in str(e).lower():
            print("  Model does not support system role — merging into user message")
            messages = [
                {"role": "user", "content": system_prompt + "\n" + user_message},
                {"role": "assistant", "content": response},
            ]
            reference_ids = _apply_template(tokenizer, messages)
        else:
            raise

    # Decode full sequence for content search. Using decode() on the full
    # sequence (not individual tokens) handles SentencePiece leading-space
    # markers correctly, unlike per-token decode + concatenation.
    full_decoded = tokenizer.decode(reference_ids, skip_special_tokens=False)

    # Find content pieces in decoded text via string search
    boundaries: Dict[str, Tuple[int, int]] = {}
    search_from = 0

    for piece_name, piece_text in [
        ("system_prompt", system_prompt),
        ("user_message", user_message),
        ("response", response),
    ]:
        char_pos = full_decoded.find(piece_text, search_from)
        if char_pos >= 0:
            char_end = char_pos + len(piece_text)
            tok_s = _char_to_token_bisect(tokenizer, reference_ids, char_pos)
            tok_e = _char_to_token_bisect(tokenizer, reference_ids, char_end - 1) + 1
            boundaries[piece_name] = (tok_s, tok_e)
            search_from = char_end
        else:
            print(f"WARNING: Could not locate {piece_name} in decoded text")

    # chat_template = everything not in content pieces
    content_indices = set()
    for start, end in boundaries.values():
        content_indices.update(range(start, end))

    chat_indices = set(range(len(reference_ids))) - content_indices
    if chat_indices:
        boundaries["chat_template"] = (min(chat_indices), max(chat_indices) + 1)

    return list(reference_ids), boundaries


def _apply_template(tokenizer, messages: list) -> List[int]:
    """Apply chat template and extract token IDs."""
    ref_result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    if hasattr(ref_result, "keys"):
        reference_ids = ref_result["input_ids"]
        # Handle nested lists (batched output) or Encoding objects
        if isinstance(reference_ids, list) and reference_ids and isinstance(reference_ids[0], list):
            reference_ids = reference_ids[0]
        elif not isinstance(reference_ids, list):
            reference_ids = list(reference_ids)
    else:
        reference_ids = list(ref_result)
    # Ensure plain list of ints
    if reference_ids and not isinstance(reference_ids[0], int):
        reference_ids = [int(x) for x in reference_ids]
    return reference_ids


def _char_to_token_bisect(
    tokenizer, token_ids: List[int], target_char: int
) -> int:
    """Binary search for the token index containing target_char in decoded text.

    Uses progressive prefix decoding: decode(token_ids[:n]) gives the text
    up through token n-1. Binary search finds the smallest n where the
    decoded prefix length exceeds target_char.
    """
    lo, hi = 0, len(token_ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        prefix_len = len(tokenizer.decode(token_ids[:mid + 1], skip_special_tokens=False))
        if prefix_len <= target_char:
            lo = mid + 1
        else:
            hi = mid
    return lo


def resolve_char_regions_to_tokens(
    tokenizer,
    text: str,
    text_tokens: List[int],
    char_regions: Dict[str, Dict[str, int]],
    global_offset: int,
) -> Dict[str, Dict]:
    """Map character-level regions to token positions via cumulative decode.

    This avoids BPE boundary issues with large regions that fail subsequence
    matching. Builds a char->token map by decoding each token to get its
    character length.
    """
    cum_chars = [0]
    for tok_id in text_tokens:
        tok_text = tokenizer.decode([tok_id])
        cum_chars.append(cum_chars[-1] + len(tok_text))

    def char_to_token(char_pos: int) -> int:
        idx = bisect.bisect_right(cum_chars, char_pos) - 1
        return max(0, min(idx, len(text_tokens) - 1))

    decoded_len = cum_chars[-1]
    if abs(decoded_len - len(text)) > len(text) * 0.05:
        print(f"  WARNING: Decoded text length ({decoded_len}) differs from original ({len(text)}) by >{5}%")

    resolved = {}
    for region_name, bounds in char_regions.items():
        char_start = bounds["char_start"]
        char_end = bounds["char_end"]

        tok_start_local = char_to_token(char_start)
        tok_end_local = char_to_token(char_end - 1) + 1 if char_end > char_start else tok_start_local

        tok_start_global = global_offset + tok_start_local
        tok_end_global = global_offset + tok_end_local
        n_tokens = tok_end_local - tok_start_local

        if n_tokens <= 0:
            print(f"  WARNING: Region '{region_name}' resolved to 0 tokens")
            continue

        resolved[region_name] = {
            "tok_start": tok_start_global,
            "tok_end": tok_end_global,
            "n_tokens": n_tokens,
        }

    return resolved


def build_full_region_map(
    tokenizer,
    token_ids: List[int],
    piece_boundaries: Dict[str, Tuple[int, int]],
    system_prompt: str,
    user_message: str,
    response: str,
    system_char_regions: Dict[str, Dict[str, int]],
    user_char_regions: Dict[str, Dict[str, int]],
    response_char_regions: Dict[str, Dict[str, int]],
) -> Dict[str, Dict]:
    """Build complete token-level region map from all character-level annotations."""
    region_map: Dict[str, Dict] = {}

    # Top-level pieces
    for piece_name in ["system_prompt", "user_message", "response", "chat_template"]:
        if piece_name in piece_boundaries:
            start, end = piece_boundaries[piece_name]
            n_tokens = end - start
            # For chat_template, count actual non-content tokens
            region_map[piece_name] = {
                "tok_start": start,
                "tok_end": end,
                "n_tokens": n_tokens,
            }

    # System prompt sub-regions
    sys_piece = piece_boundaries.get("system_prompt")
    if sys_piece and system_char_regions:
        sys_start, sys_end = sys_piece
        sys_tokens = token_ids[sys_start:sys_end]
        resolved = resolve_char_regions_to_tokens(
            tokenizer, system_prompt, sys_tokens, system_char_regions, sys_start
        )
        region_map.update(resolved)

    # User message sub-regions
    usr_piece = piece_boundaries.get("user_message")
    if usr_piece and user_char_regions:
        usr_start, usr_end = usr_piece
        usr_tokens = token_ids[usr_start:usr_end]
        resolved = resolve_char_regions_to_tokens(
            tokenizer, user_message, usr_tokens, user_char_regions, usr_start
        )
        region_map.update(resolved)

    # Response sub-regions
    resp_piece = piece_boundaries.get("response")
    if resp_piece and response_char_regions:
        resp_start, resp_end = resp_piece
        resp_tokens = token_ids[resp_start:resp_end]
        resolved = resolve_char_regions_to_tokens(
            tokenizer, response, resp_tokens, response_char_regions, resp_start
        )
        region_map.update(resolved)

    return region_map


# ============================================================================
# QUERY POSITION RESOLUTION
# ============================================================================

def resolve_query_positions(
    tokenizer,
    token_ids: List[int],
    piece_boundaries: Dict[str, Tuple[int, int]],
    position_defs: Optional[Dict] = None,
) -> Dict[str, int]:
    """Resolve query positions within the token sequence.

    Default positions:
    - terminal: last token of user message

    Additional positions from position_defs in test_cases.json:
    - "last_token": last token of user message
    - {"after_text": "..."}: first token after the specified text in response
    - {"at_text": "..."}: token at the specified text in response
    """
    positions: Dict[str, int] = {}

    # terminal: last token of user message
    usr_piece = piece_boundaries.get("user_message")
    if usr_piece:
        _, usr_end = usr_piece
        positions["terminal"] = usr_end - 1

    # Custom positions from config
    if position_defs:
        resp_piece = piece_boundaries.get("response")
        resp_start = resp_piece[0] if resp_piece else 0
        resp_end = resp_piece[1] if resp_piece else len(token_ids)

        for pos_name, pos_def in position_defs.items():
            if pos_name == "terminal":
                continue  # already handled

            if pos_def == "last_token":
                if usr_piece:
                    positions[pos_name] = usr_piece[1] - 1

            elif isinstance(pos_def, dict):
                if "after_text" in pos_def:
                    text = pos_def["after_text"]
                    text_tokens = tokenizer.encode(text, add_special_tokens=False)
                    resp_tokens = token_ids[resp_start:resp_end]
                    idx = _find_subsequence(resp_tokens, text_tokens)
                    if idx >= 0:
                        content_start = resp_start + idx + len(text_tokens)
                        # Skip whitespace tokens
                        while content_start < resp_end:
                            tok_text = tokenizer.decode([token_ids[content_start]]).strip()
                            if tok_text:
                                break
                            content_start += 1
                        if content_start < resp_end:
                            positions[pos_name] = content_start

                elif "at_text" in pos_def:
                    text = pos_def["at_text"]
                    text_tokens = tokenizer.encode(text, add_special_tokens=False)
                    resp_tokens = token_ids[resp_start:resp_end]
                    idx = _find_subsequence(resp_tokens, text_tokens)
                    if idx >= 0:
                        positions[pos_name] = resp_start + idx

    return positions


# ============================================================================
# FORWARD HOOKS
# ============================================================================

class ResidualCache:
    """Stores residual stream vectors at query positions across all layers."""

    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _key(layer: int, pos_name: str) -> str:
        return f"resid_L{layer}_{pos_name}"

    def make_hook(self, layer_idx: int, query_positions: Dict[str, int]):
        cache = self.cache

        def hook_fn(module, input, output):
            h = output[0]
            if h.dim() == 2:
                h = h.unsqueeze(0)
            for pos_name, pos_idx in query_positions.items():
                if pos_idx < h.shape[1]:
                    cache[ResidualCache._key(layer_idx, pos_name)] = h[0, pos_idx, :].float().cpu()

        return hook_fn

    def get(self, layer: int, position_name: str) -> Optional[torch.Tensor]:
        return self.cache.get(self._key(layer, position_name))


class AttentionCache:
    """Extracts per-region attention weights at query positions, discards full matrix."""

    def __init__(self):
        self.cache: Dict[str, Dict[str, List[float]]] = {}
        self.per_token_cache: Dict[str, List[float]] = {}

    @staticmethod
    def _key(layer: int, pos_name: str) -> str:
        return f"attn_L{layer}_{pos_name}"

    def make_hook(
        self,
        layer_idx: int,
        query_positions: Dict[str, int],
        region_map: Dict[str, Dict],
        capture_per_token: bool = True,
    ):
        cache = self.cache
        per_token_cache = self.per_token_cache

        def hook_fn(module, input, output):
            if len(output) < 2 or output[1] is None:
                return output

            A = output[1]
            if A.dim() == 3:
                A = A.unsqueeze(0)

            for pos_name, pos_idx in query_positions.items():
                if pos_idx >= A.shape[2]:
                    continue

                row = A[0, :, pos_idx, :].float()

                region_weights = {}
                for region_name, region_info in region_map.items():
                    tok_start = region_info["tok_start"]
                    tok_end = region_info["tok_end"]
                    if tok_end <= A.shape[3]:
                        weight = row[:, tok_start:tok_end].sum(dim=1).cpu().tolist()
                        region_weights[region_name] = weight

                cache[AttentionCache._key(layer_idx, pos_name)] = region_weights

                if capture_per_token:
                    per_tok = row.mean(dim=0).cpu().tolist()
                    per_token_cache[AttentionCache._key(layer_idx, pos_name)] = per_tok

            # Replace attention weights with None to free ~2.2GB per layer
            return (output[0], None) + output[2:]

        return hook_fn

    def get(self, layer: int, position_name: str) -> Optional[Dict[str, List[float]]]:
        return self.cache.get(self._key(layer, position_name))

    def get_per_token(self, layer: int, position_name: str) -> Optional[List[float]]:
        return self.per_token_cache.get(self._key(layer, position_name))


# ============================================================================
# LOGIT LENS
# ============================================================================

def compute_logit_lens(
    norm,
    lm_head,
    residual_cache: ResidualCache,
    query_positions: Dict[str, int],
    tracked_token_ids: Dict[str, int],
    num_layers: int,
    top_k: int = 50,
    tokenizer=None,
) -> Dict[str, List[Dict]]:
    """Project residual stream through final norm + lm_head at each layer."""
    device = next(lm_head.parameters()).device

    results: Dict[str, List[Dict]] = {}

    for pos_name in query_positions:
        layer_results = []

        for layer_idx in range(num_layers):
            h = residual_cache.get(layer_idx, pos_name)
            if h is None:
                continue

            with torch.no_grad():
                model_dtype = next(lm_head.parameters()).dtype
                h_device = h.unsqueeze(0).unsqueeze(0).to(device=device, dtype=model_dtype)
                h_normed = norm(h_device)
                logits = lm_head(h_normed).squeeze().float()
                probs = F.softmax(logits, dim=-1)

            topk_logits, topk_indices = torch.topk(logits, top_k)
            topk_probs = probs[topk_indices]

            top_k_list = []
            for rank, (tok_id, logit_val, prob_val) in enumerate(
                zip(topk_indices.tolist(), topk_logits.tolist(), topk_probs.tolist())
            ):
                tok_str = tokenizer.decode([tok_id]) if tokenizer else str(tok_id)
                top_k_list.append({
                    "token": tok_str,
                    "token_id": tok_id,
                    "logit": round(logit_val, 4),
                    "prob": round(prob_val, 6),
                    "rank": rank + 1,
                })

            tracked = {}
            for tok_str, tok_id in tracked_token_ids.items():
                tok_logit = logits[tok_id].item()
                tok_prob = probs[tok_id].item()
                tok_rank = (logits > tok_logit).sum().item() + 1
                tracked[tok_str] = {
                    "token_id": tok_id,
                    "logit": round(tok_logit, 4),
                    "prob": round(tok_prob, 6),
                    "rank": int(tok_rank),
                }

            layer_results.append({
                "layer": layer_idx,
                "top_k": top_k_list,
                "tracked": tracked,
            })

        results[pos_name] = layer_results

    return results


# ============================================================================
# ATTENTION AGGREGATION
# ============================================================================

def aggregate_attention(
    attention_cache: AttentionCache,
    query_positions: Dict[str, int],
    region_map: Dict[str, Dict],
    num_layers: int,
    num_query_heads: int,
) -> Dict[str, Dict]:
    """Aggregate raw per-head attention into per-region means and head specialization."""
    results: Dict[str, Dict] = {}

    for pos_name in query_positions:
        per_layer = []

        for layer_idx in range(num_layers):
            attn_data = attention_cache.get(layer_idx, pos_name)
            if attn_data is None:
                continue

            per_region_mean = {}
            for region_name, head_weights in attn_data.items():
                if head_weights:
                    per_region_mean[region_name] = round(
                        sum(head_weights) / len(head_weights), 6
                    )

            head_max_region = []
            for head_idx in range(num_query_heads):
                best_region = None
                best_weight = -1.0
                for region_name, head_weights in attn_data.items():
                    if head_idx < len(head_weights) and head_weights[head_idx] > best_weight:
                        best_weight = head_weights[head_idx]
                        best_region = region_name
                if best_region is not None:
                    head_max_region.append({
                        "head": head_idx,
                        "region": best_region,
                        "weight": round(best_weight, 6),
                    })

            per_layer.append({
                "layer": layer_idx,
                "per_region_mean": per_region_mean,
                "head_max_region": head_max_region,
            })

        results[pos_name] = {"per_layer": per_layer}

    return results


def collect_per_token_attention(
    attention_cache: AttentionCache,
    query_positions: Dict[str, int],
    num_layers: int,
) -> Dict[str, Dict]:
    """Collect per-token attention weights for heatmap visualization."""
    results: Dict[str, Dict] = {}

    for pos_name in query_positions:
        per_layer = []
        for layer_idx in range(num_layers):
            weights = attention_cache.get_per_token(layer_idx, pos_name)
            if weights is not None:
                per_layer.append({"layer": layer_idx, "weights": weights})
        results[pos_name] = {"per_layer": per_layer}

    return results


# ============================================================================
# VERIFICATION
# ============================================================================

def run_verifications(
    token_ids: List[int],
    region_map: Dict[str, Dict],
    logit_lens_results: Dict[str, List[Dict]],
    peak_memory_gb: float,
):
    """Run integrity checks on the analysis output."""
    issues = []

    total_tokens = len(token_ids)
    covered = set()
    for region_name, info in region_map.items():
        for idx in range(info["tok_start"], info["tok_end"]):
            covered.add(idx)
    coverage = len(covered) / total_tokens if total_tokens > 0 else 0
    if coverage < 0.8:
        issues.append(f"Region coverage only {coverage:.1%} of {total_tokens} tokens")
    print(f"  Region coverage: {len(covered)}/{total_tokens} tokens ({coverage:.1%})")

    for pos_name, layers in logit_lens_results.items():
        if layers:
            final = layers[-1]
            if final["top_k"]:
                top_token = final["top_k"][0]["token"]
                print(f"  Final layer top token at '{pos_name}': '{top_token}' (prob={final['top_k'][0]['prob']:.4f})")

    if peak_memory_gb > 75:
        issues.append(f"Peak GPU memory {peak_memory_gb:.1f} GB exceeds 75 GB safety margin")
    print(f"  Peak GPU memory: {peak_memory_gb:.1f} GB")

    if issues:
        print(f"\n  VERIFICATION ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All verifications passed")

    return issues


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================

def analyze_case(
    model,
    tokenizer,
    model_config: dict,
    case: Dict,
    system_prompt: str,
    system_regions: Dict[str, Dict[str, int]],
    top_k: int,
    requested_positions: Optional[List[str]],
    tracked_tokens: List[str],
    capture_per_token: bool = True,
    position_defs: Optional[Dict] = None,
) -> Dict:
    """Run full MI analysis on a single test case."""
    MAX_CASE_CHARS = 20_000

    num_layers = model_config["num_layers"]
    num_query_heads = model_config["num_query_heads"]
    attention_modules = model_config["attention_modules"]
    layer_modules = model_config["layer_modules"]
    norm = model_config["norm"]
    lm_head = model_config["lm_head"]
    model_name = model_config["model_name"]

    case_id = case["id"]
    user_message = case["user_message"]
    response = case["response"]
    user_regions = case.get("user_regions", {})
    response_regions = case.get("response_regions", {})

    total_chars = len(system_prompt) + len(user_message) + len(response)
    if total_chars > MAX_CASE_CHARS:
        print(f"\nSKIPPING {case_id}: {total_chars} chars exceeds {MAX_CASE_CHARS} limit")
        return None

    print(f"\n{'=' * 70}")
    print(f"Analyzing: {case_id}")
    print(f"{'=' * 70}")

    device = next(model.parameters()).device
    start_time = time.monotonic()

    # Step 1: Build token sequence
    print("Building token sequence...")
    token_ids, piece_boundaries = build_chat_tokens(
        tokenizer, system_prompt, user_message, response
    )
    seq_len = len(token_ids)
    print(f"  Sequence length: {seq_len} tokens")

    # Step 2: Build token-level region map
    print("Resolving region annotations to token positions...")
    region_map = build_full_region_map(
        tokenizer, token_ids, piece_boundaries,
        system_prompt, user_message, response,
        system_regions, user_regions, response_regions,
    )

    for name, info in sorted(region_map.items(), key=lambda x: x[1]["tok_start"]):
        print(f"  {name:25s} tokens {info['tok_start']:5d}-{info['tok_end']:5d} ({info['n_tokens']:4d} tokens)")

    # Step 3: Resolve query positions
    print("Resolving query positions...")
    query_positions = resolve_query_positions(
        tokenizer, token_ids, piece_boundaries, position_defs
    )

    if requested_positions:
        query_positions = {k: v for k, v in query_positions.items() if k in requested_positions}

    for name, pos in query_positions.items():
        tok_text = tokenizer.decode([token_ids[pos]]) if pos < len(token_ids) else "?"
        print(f"  {name:20s} position {pos:5d} (token: '{tok_text}')")

    if not query_positions:
        print("WARNING: No query positions resolved")
        return {"error": "no_query_positions", "case_id": case_id}

    # Step 4: Resolve tracked token IDs
    tracked_token_ids = {}
    for tok_str in tracked_tokens:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if ids:
            tracked_token_ids[tok_str] = ids[0]

    # Step 5: Register hooks
    print("Registering forward hooks...")
    residual_cache = ResidualCache()
    attention_cache = AttentionCache()
    handles = []

    for layer_idx, layer_module in layer_modules:
        h = layer_module.register_forward_hook(
            residual_cache.make_hook(layer_idx, query_positions)
        )
        handles.append(h)

    for layer_idx, attn_module in attention_modules:
        # Hook self_attn directly (not the decoder layer) to avoid
        # accelerate's AlignDevicesHook ordering issues
        h = attn_module.register_forward_hook(
            attention_cache.make_hook(layer_idx, query_positions, region_map, capture_per_token)
        )
        handles.append(h)

    print(f"  Registered {len(handles)} hooks across {num_layers} layers")

    # Step 6: Forward pass
    print("Running forward pass...")
    input_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

    torch.cuda.reset_peak_memory_stats()
    forward_start = time.monotonic()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    del outputs

    forward_seconds = time.monotonic() - forward_start
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    print(f"  Forward pass: {forward_seconds:.1f}s, peak GPU: {peak_memory_gb:.1f} GB")

    for h in handles:
        h.remove()

    # Step 7: Logit lens
    print("Computing logit lens...")
    logit_lens_results = compute_logit_lens(
        norm, lm_head, residual_cache, query_positions,
        tracked_token_ids, num_layers, top_k=top_k, tokenizer=tokenizer,
    )

    # Step 8: Aggregate attention
    print("Aggregating attention patterns...")
    attention_results = aggregate_attention(
        attention_cache, query_positions, region_map, num_layers, num_query_heads
    )

    # Step 8b: Per-token attention
    per_token_results = None
    token_labels = None
    if capture_per_token:
        print("Collecting per-token attention for heatmap...")
        per_token_results = collect_per_token_attention(attention_cache, query_positions, num_layers)
        token_labels = [tokenizer.decode([tok_id]) for tok_id in token_ids]

    # Cleanup
    del residual_cache, attention_cache
    del input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    # Step 9: Verification
    print("Running verifications...")
    issues = run_verifications(token_ids, region_map, logit_lens_results, peak_memory_gb)

    elapsed = time.monotonic() - start_time
    print(f"\nCase {case_id} complete in {elapsed:.1f}s")

    result = {
        "metadata": {
            "model": model_name,
            "dtype": "float16",
            "attn_implementation": "eager",
            "case_id": case_id,
            "total_tokens": seq_len,
            "peak_gpu_memory_gb": round(peak_memory_gb, 1),
            "forward_pass_seconds": round(forward_seconds, 1),
            "total_seconds": round(elapsed, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verification_issues": issues,
        },
        "region_map": region_map,
        "query_positions": query_positions,
        "logit_lens": logit_lens_results,
        "attention": attention_results,
    }

    if per_token_results is not None:
        result["per_token_attention"] = per_token_results
    if token_labels is not None:
        result["token_labels"] = token_labels

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mechanistic interpretability analysis for HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to test_cases.json",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for per-case result JSON files",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Number of top tokens in logit lens (default: 50)",
    )
    parser.add_argument(
        "--query-positions", default=None,
        help="Comma-separated query positions to analyze (default: all found)",
    )
    parser.add_argument(
        "--cases", default=None,
        help="Comma-separated case IDs to analyze (default: all)",
    )
    parser.add_argument(
        "--no-per-token", action="store_true",
        help="Skip per-token attention capture (saves memory in output)",
    )
    parser.add_argument(
        "--tracked-tokens", nargs="*", default=[],
        help="Tokens to track in logit lens (e.g., '<' 'folder_a')",
    )
    args = parser.parse_args()

    requested_positions = None
    if args.query_positions:
        requested_positions = [p.strip() for p in args.query_positions.split(",")]

    requested_cases = None
    if args.cases:
        requested_cases = set(c.strip() for c in args.cases.split(","))

    # Load test cases
    print("Loading test cases...")
    input_path = Path(args.input)
    with open(input_path) as f:
        test_data = json.load(f)

    system_prompt = test_data["system_prompt"]
    system_regions = test_data["system_regions"]
    cases = test_data["cases"]
    position_defs = test_data.get("query_positions", {})

    # Tracked tokens: CLI args override, then test_cases.json, then empty
    tracked_tokens = args.tracked_tokens
    if not tracked_tokens:
        tracked_tokens = test_data.get("tracked_tokens", [])

    if requested_cases:
        cases = [c for c in cases if c["id"] in requested_cases]

    print(f"  System prompt: {len(system_prompt)} chars, {len(system_regions)} regions")
    print(f"  Cases to analyze: {len(cases)}")
    if tracked_tokens:
        print(f"  Tracked tokens: {tracked_tokens}")

    if not cases:
        print("ERROR: No cases to analyze")
        return

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.eval()

    model_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  Model loaded: {model_memory:.1f} GB GPU memory")

    # Auto-discover model architecture
    print("Discovering model architecture...")
    model_config = _discover_model_config(model)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    for i, case in enumerate(cases):
        print(f"\n{'#' * 70}")
        print(f"Case {i + 1}/{len(cases)}: {case['id']}")
        print(f"{'#' * 70}")

        result = analyze_case(
            model, tokenizer, model_config, case,
            system_prompt, system_regions,
            top_k=args.top_k,
            requested_positions=requested_positions,
            tracked_tokens=tracked_tokens,
            capture_per_token=not args.no_per_token,
            position_defs=position_defs,
        )

        if result is None:
            continue

        output_path = output_dir / f"{case['id']}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Written: {output_path}")

        del result

    print(f"\n{'=' * 70}")
    print(f"Analysis complete. Results in: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
