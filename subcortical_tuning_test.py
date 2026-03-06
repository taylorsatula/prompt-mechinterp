#!/usr/bin/env python3
"""
Subcortical Tuning Test: Evaluate query expansion, entity extraction, passage retention.

Grabs random 7-message segments, runs a 2-turn simulation per sample:
Turn 1 (warmup) generates query expansion with no memories, retrieves real
memories via vector search. Turn 2 (measured) generates with retrieved memories.
Read-only — no database writes.

Uses MIRA app-layer services (not direct psycopg3):
- VectorOps.find_similar_memories() for memory retrieval between turns
- config/prompts/subcortical_system.txt + subcortical_user.txt
- Standalone message formatting and response parsing (does not call SubcorticalLayer)

Usage:
    python scripts/tuning_harnesses/subcortical_tuning_test.py --seed 1
    python scripts/tuning_harnesses/subcortical_tuning_test.py --seed 3 --model openai/gpt-oss-20b

Output (with --save):
    - data/tuning_test_results/subcortical/YYYY-MM-DD/md/subcortical_HHMMSS.md
    - data/tuning_test_results/subcortical/YYYY-MM-DD/json/subcortical_HHMMSS.json
"""
import argparse
import sys
sys.path.insert(0, '.')

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Any
from uuid import UUID

import numpy as np

from auth.database import AuthDatabase
from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
from clients.vault_client import get_api_key
from lt_memory.db_access import LTMemoryDB
from lt_memory.vector_ops import VectorOps
from utils.database_session_manager import get_shared_session_manager
from utils.user_context import set_current_user_id
from utils.timezone_utils import utc_now


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

LLM_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-3-flash-preview"
LLM_API_KEY_NAME = "openrouter_key"
LLM_MAX_TOKENS = 2000


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MessageRecord:
    """Database message record."""
    id: UUID
    continuum_id: UUID
    user_id: str
    role: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """Memory record for formatting."""
    id: str
    text: str
    importance_score: float


@dataclass
class SubcorticalOutput:
    """Parsed subcortical output."""
    query_expansion: str
    entities: List[str]
    retained_ids: Set[str]
    raw_response: str


@dataclass
class SampleResult:
    """Result for a single sample."""
    sample_num: int
    messages: List[MessageRecord]
    current_message: str

    turn1_query_expansion: str
    turn1_memories: List[MemoryRecord]
    turn2_output: SubcorticalOutput
    assembled_user_prompt: str = ""


# ============================================================================
# PROMPT LOADING
# ============================================================================

def load_prompts(
    system_path: str = None,
    user_path: str = None,
) -> tuple:
    """Load prompts from files. Uses production defaults if paths not specified."""
    system_path = Path(system_path) if system_path else Path("config/prompts/subcortical_system.txt")
    user_path = Path(user_path) if user_path else Path("config/prompts/subcortical_user.txt")

    with open(system_path, 'r') as f:
        system_prompt = f.read()
    with open(user_path, 'r') as f:
        user_template = f.read()

    return system_prompt, user_template


# ============================================================================
# FORMATTING FUNCTIONS
# ============================================================================

def importance_to_dots(importance_score: float) -> str:
    """Convert importance score (0.0-1.0) to 5-dot visual indicator."""
    score = max(0.0, min(1.0, importance_score))
    filled = int(score * 5) + (1 if score > 0 else 0)
    filled = min(5, max(1, filled)) if score > 0 else 1
    return "●" * filled + "○" * (5 - filled)


def format_memory_id(memory_id: str) -> str:
    """Format UUID to mem_XXXXXXXX format."""
    short_id = memory_id.replace('-', '')[:8]
    return f"mem_{short_id}"


_MIRA_TAG_PATTERN = re.compile(r'<mira:[^>]*>.*?</mira:[^>]*>|<mira:[^/]*/\s*>', re.DOTALL)


def format_conversation(messages: List[MessageRecord]) -> str:
    """Format conversation as XML turns with timestamps.

    Matches production behavior (SubcorticalLayer._format_recent_turns):
    - Strips <mira:*> internal tags from assistant content
    - Truncates each message to 2000 chars
    """
    lines = []
    for msg in messages:
        time_str = msg.created_at.strftime("%H:%M")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Strip <mira:*> internal tags (emotion emojis, memory refs) — rare-token
        # attention sinks that waste budget in subcortical processing.
        if msg.role == "assistant":
            content = _MIRA_TAG_PATTERN.sub('', content)
        content = content[:2000]
        lines.append(f'<turn speaker="{msg.role}" time="{time_str}">{content}</turn>')
    return "\n".join(lines)


def format_conversation_decay(messages: List[MessageRecord]) -> tuple:
    """Format conversation with decay: recent turns verbose, older turns as keywords.

    Returns (older_context, recent_turns) for the conv_decay variant template
    which uses {older_context} and {recent_turns} placeholders.
    """
    if len(messages) <= 2:
        # Too few for decay — all recent
        return "", format_conversation(messages)

    # Most recent turn pair stays verbose
    recent = messages[-2:]
    older = messages[:-2]

    # Recent: full XML turns (with mira tag stripping via format_conversation)
    recent_text = format_conversation(recent)

    # Older: extract nouns/topics as keyword summary
    import re
    keywords = set()
    for msg in older:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Extract capitalized words (likely proper nouns/topics)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        keywords.update(words[:5])  # cap per message
        # Also grab quoted terms and significant nouns
        quoted = re.findall(r'"([^"]+)"', content)
        keywords.update(q for q in quoted if len(q) < 40)
    # Fallback: if no caps found, take first few words of each message
    if not keywords:
        for msg in older:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            words = content.split()[:6]
            keywords.update(w for w in words if len(w) > 3)

    older_text = ", ".join(sorted(keywords)[:15])
    return older_text, recent_text


def format_memories(memories: List[MemoryRecord], truncate_words: int = 0) -> str:
    """Format memories as passage lines (template wraps).

    truncate_words: if > 0, truncate each passage text to first N words.
    """
    if not memories:
        return ""
    lines = []
    for m in memories:
        formatted_id = format_memory_id(m.id)
        dots = importance_to_dots(m.importance_score)
        text = m.text
        if truncate_words > 0:
            words = text.split()
            if len(words) > truncate_words:
                text = " ".join(words[:truncate_words]) + "..."
        lines.append(f"{formatted_id} [{dots}] - {text}")
    return "\n".join(lines)


def build_user_prompt(
    template: str,
    conversation: str,
    current_msg: str,
    memories_block: str,
    older_context: str = "",
    recent_turns: str = "",
) -> str:
    """Build user prompt from template.

    For conv_decay variant, uses {older_context} and {recent_turns} instead
    of {conversation_turns}.
    """
    result = template.replace(
        "{conversation_turns}", conversation
    ).replace(
        "{user_message}", current_msg
    ).replace(
        "{previous_memories}", memories_block
    )
    # Conv decay variant placeholders
    if "{older_context}" in result:
        result = result.replace("{older_context}", older_context)
    if "{recent_turns}" in result:
        result = result.replace("{recent_turns}", recent_turns)
    return result


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_response(response_text: str) -> SubcorticalOutput:
    """Parse subcortical response."""
    # Extract query_expansion from <query_expansion>
    query_expansion = ""
    fp_match = re.search(
        r'<query_expansion>(.*?)</query_expansion>',
        response_text,
        re.DOTALL
    )
    if fp_match:
        query_expansion = fp_match.group(1).strip()

    # Extract entities from <ne> tags
    entities = []
    ent_match = re.search(
        r'<entities>(.*?)</entities>',
        response_text,
        re.DOTALL
    )
    if ent_match:
        block = ent_match.group(1).strip()
        if block.lower() != "none":
            for match in re.finditer(r'<ne[^>]*>(.*?)</ne>', block, re.DOTALL):
                name = match.group(1).strip()
                if name and name.lower() != "none":
                    entities.append(name)

    # Extract retained IDs from <passage id="mem_xxx">
    retained_ids: Set[str] = set()
    for match in re.finditer(
        r'<passage\s+id="mem_([a-fA-F0-9]{8})"',
        response_text,
        re.IGNORECASE
    ):
        retained_ids.add(match.group(1).lower())

    return SubcorticalOutput(
        query_expansion=query_expansion,
        entities=entities,
        retained_ids=retained_ids,
        raw_response=response_text
    )


# ============================================================================
# CORE SERVICES
# ============================================================================

def setup_user_context() -> str:
    """Set user context for RLS using known test user UUIDs."""
    test_user_ids = [
        "999eadba-542c-4654-a4ee-dbb8ac2ee2cd",
        "46583e05-1702-49f6-86aa-7eadfc01c568",
    ]

    auth_db = AuthDatabase()
    for user_id in test_user_ids:
        user_record = auth_db.get_user_by_id(user_id)
        if user_record:
            set_current_user_id(user_id)
            return user_id

    raise RuntimeError(f"Test user not found with either UUID: {test_user_ids}")


def setup_services():
    """Initialize real services for testing."""
    session_manager = get_shared_session_manager()
    lt_memory_db = LTMemoryDB(session_manager)
    embeddings = get_hybrid_embeddings_provider(cache_enabled=False)
    vector_ops = VectorOps(embeddings, lt_memory_db)

    return {
        "session_manager": session_manager,
        "lt_memory_db": lt_memory_db,
        "embeddings": embeddings,
        "vector_ops": vector_ops,
    }


def get_random_segments(session_manager, user_id: str, limit: int = 20) -> List[List[MessageRecord]]:
    """
    Get random 7-message segments from the messages table.

    Returns list of 7-message segments for testing.
    Picks multiple non-overlapping segments from available continuums.

    IMPORTANT: The 7th message (current turn) must be a user message,
    since subcortical layer generation only runs when users send messages.
    """
    with session_manager.get_session(user_id) as session:
        # Get all valid messages ordered by time
        messages = session.execute_query("""
            SELECT id, continuum_id, user_id, role, content, created_at, metadata
            FROM messages
            WHERE role IN ('user', 'assistant')
              AND (metadata->>'is_segment_boundary' IS NULL
                   OR metadata->>'is_segment_boundary' = 'false')
              AND (metadata->>'system_notification' IS NULL
                   OR metadata->>'system_notification' = 'false')
            ORDER BY created_at ASC
        """)

        if len(messages) < 7:
            raise RuntimeError(f"Not enough messages: {len(messages)} (need at least 7)")

        # Find all valid starting points where the 7th message is a user message
        # This mirrors production: subcortical layer only fires on user turns
        valid_starts = []
        for i in range(len(messages) - 6):
            if messages[i + 6]['role'] == 'user':
                valid_starts.append(i)

        if not valid_starts:
            raise RuntimeError("No valid 7-message segments ending with user message")

        print(f"  Found {len(valid_starts)} valid start positions (ending with user message)")

        # Select non-overlapping segments from valid starts
        random.shuffle(valid_starts)

        selected_starts = []
        for candidate in valid_starts:
            if len(selected_starts) >= limit:
                break
            # Check it doesn't overlap with already selected segments
            if all(abs(candidate - s) >= 7 for s in selected_starts):
                selected_starts.append(candidate)

        segments = []
        for start_idx in selected_starts:
            segment_messages = []
            for row in messages[start_idx:start_idx + 7]:
                content = row['content']
                if isinstance(content, dict):
                    content = json.dumps(content)

                segment_messages.append(MessageRecord(
                    id=row['id'],
                    continuum_id=row['continuum_id'],
                    user_id=row['user_id'],
                    role=row['role'],
                    content=content,
                    created_at=row['created_at'],
                    metadata=row.get('metadata') or {}
                ))
            segments.append(segment_messages)

    return segments


def retrieve_real_memories(
    query_expansion: str,
    vector_ops: VectorOps,
    limit: int = 5
) -> List[MemoryRecord]:
    """Use query expansion to retrieve REAL memories via vector search."""
    results = vector_ops.find_similar_memories(
        query=query_expansion,
        limit=limit,
        similarity_threshold=0.3,
        min_importance=0.05
    )

    return [
        MemoryRecord(
            id=str(m.id),
            text=m.text,
            importance_score=m.importance_score
        )
        for m in results
    ]


def run_llm_call(
    system_prompt: str,
    user_prompt: str,
    model: str = LLM_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    endpoint: str = LLM_ENDPOINT,
    api_key_name: str = LLM_API_KEY_NAME,
    api_key_direct: str = None,
) -> tuple:
    """Direct POST to LLM API for subcortical generation.

    Returns:
        Tuple of (response_text, stop_reason, elapsed_ms)
    """
    import requests

    api_key = api_key_direct if api_key_direct else get_api_key(api_key_name)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.monotonic()
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    elapsed_ms = int((time.monotonic() - start) * 1000)

    data = resp.json()
    raw_text = data["choices"][0]["message"]["content"]
    stop_reason = data["choices"][0].get("finish_reason", "unknown")

    return raw_text, stop_reason, elapsed_ms


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_test(
    sample_num: int,
    messages: List[MessageRecord],
    services: Dict[str, Any],
    system_prompt: str,
    user_template: str,
    model: str = LLM_MODEL,
    endpoint: str = LLM_ENDPOINT,
    api_key_name: str = LLM_API_KEY_NAME,
    api_key_direct: str = None,
    quiet: bool = False,
    context_pairs: int = 6,
    conversation_decay: bool = False,
    truncate_passages: int = 0,
) -> SampleResult:
    """
    Run test on a single sample.

    2-turn simulation:
    - Turn 1 (warmup): Generate query expansion with no memories → retrieve real memories
    - Turn 2 (measured): Generate query expansion with retrieved memories → save output
    """
    vector_ops = services["vector_ops"]

    # Messages 1-6 for Turn 1, Messages 2-7 for Turn 2
    turn1_context = messages[:6]  # First 6 messages
    turn1_current = messages[5].content  # 6th message as "current"

    turn2_context = messages[1:7]  # Messages 2-7
    turn2_current = messages[6].content  # 7th message as "current"

    # Turn 1 (warmup) - no memories
    if not quiet:
        print(f"  Turn 1: Generating query expansion...")

    # Apply context_pairs limit: take last N context messages before current
    turn1_conv_messages = turn1_context[:-1]
    if context_pairs < 6:
        turn1_conv_messages = turn1_conv_messages[-context_pairs:]

    # Format conversation (with optional decay)
    older_t1, recent_t1 = "", ""
    if conversation_decay:
        older_t1, recent_t1 = format_conversation_decay(turn1_conv_messages)
        conv_t1 = format_conversation(turn1_conv_messages)  # fallback for {conversation_turns}
    else:
        conv_t1 = format_conversation(turn1_conv_messages)

    user_t1 = build_user_prompt(user_template, conv_t1, turn1_current, "",
                                older_context=older_t1, recent_turns=recent_t1)
    response_text_t1, stop_t1, elapsed_t1 = run_llm_call(system_prompt, user_t1, model=model, endpoint=endpoint, api_key_name=api_key_name, api_key_direct=api_key_direct)
    if not quiet:
        print(f"    Stop reason: {stop_t1} ({elapsed_t1}ms)")

    parsed_t1 = parse_response(response_text_t1)

    # Retrieve real memories using Turn 1 query expansion
    if not quiet:
        print(f"  Retrieving memories with: '{parsed_t1.query_expansion[:50]}...'")
    memories = retrieve_real_memories(parsed_t1.query_expansion, vector_ops)
    if not quiet:
        print(f"  Retrieved {len(memories)} memories")

    # Turn 2 (measured) - with retrieved memories
    if not quiet:
        print(f"  Turn 2: Generating query expansion with memories...")
    turn2_conv_messages = turn2_context[:-1]
    if context_pairs < 6:
        turn2_conv_messages = turn2_conv_messages[-context_pairs:]

    older_t2, recent_t2 = "", ""
    if conversation_decay:
        older_t2, recent_t2 = format_conversation_decay(turn2_conv_messages)
        conv_t2 = format_conversation(turn2_conv_messages)
    else:
        conv_t2 = format_conversation(turn2_conv_messages)

    memories_block = format_memories(memories, truncate_words=truncate_passages)
    user_t2 = build_user_prompt(user_template, conv_t2, turn2_current, memories_block,
                                older_context=older_t2, recent_turns=recent_t2)
    response_text_t2, stop_t2, elapsed_t2 = run_llm_call(system_prompt, user_t2, model=model, endpoint=endpoint, api_key_name=api_key_name, api_key_direct=api_key_direct)
    if not quiet:
        print(f"    Stop reason: {stop_t2} ({elapsed_t2}ms)")

    parsed_t2 = parse_response(response_text_t2)

    return SampleResult(
        sample_num=sample_num,
        messages=messages,
        current_message=turn2_current,
        turn1_query_expansion=parsed_t1.query_expansion,
        turn1_memories=memories,
        turn2_output=parsed_t2,
        assembled_user_prompt=user_t2,
    )


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def generate_markdown(results: List[SampleResult], timestamp: str, model: str = LLM_MODEL) -> str:
    """Generate markdown report."""
    lines = [
        "# Subcortical Layer Tuning Test Results",
        f"Generated: {timestamp}",
        f"Model: {model}",
        "",
        "---",
        ""
    ]

    for result in results:
        lines.append(f"## Sample {result.sample_num} of {len(results)}")
        lines.append("")

        # Conversation context
        lines.append("### Conversation Context")
        lines.append("```")
        for msg in result.messages[:-1]:  # All but last (current)
            time_str = msg.created_at.strftime("%H:%M")
            prefix = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"{prefix} [{time_str}]: {content}")
        lines.append("```")
        lines.append("")
        lines.append(f"**Current turn:** \"{result.current_message[:100]}{'...' if len(result.current_message) > 100 else ''}\"")
        lines.append("")

        # Turn 1 query expansion
        lines.append("### Turn 1 (warmup)")
        lines.append(f"**Query expansion:** {result.turn1_query_expansion}")
        lines.append(f"**Memories retrieved:** {len(result.turn1_memories)}")
        lines.append("")

        # Turn 2 output
        lines.append("### Turn 2 Output")
        lines.append("```xml")
        lines.append(result.turn2_output.raw_response)
        lines.append("```")
        lines.append("")

        # Quality metrics
        lines.append("### Quality Metrics")
        lines.append("")
        wc = len(result.turn2_output.query_expansion.split())
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Query expansion word count | {wc} |")
        lines.append(f"| Entities extracted | {len(result.turn2_output.entities)} |")
        lines.append(f"| Passages retained | {len(result.turn2_output.retained_ids)} |")

        if result.turn2_output.entities:
            ents = ", ".join(result.turn2_output.entities[:5])
            lines.append(f"| Entities | {ents} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary statistics
    lines.append("## Summary Statistics")
    lines.append("")

    word_counts = [len(r.turn2_output.query_expansion.split()) for r in results]
    entity_counts = [len(r.turn2_output.entities) for r in results]
    retained_counts = [len(r.turn2_output.retained_ids) for r in results]
    empty_expansions = sum(1 for r in results if not r.turn2_output.query_expansion.strip())

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Avg query expansion word count | {np.mean(word_counts):.1f} |")
    lines.append(f"| Min/Max word count | {min(word_counts)} / {max(word_counts)} |")
    lines.append(f"| Avg entities per sample | {np.mean(entity_counts):.1f} |")
    lines.append(f"| Avg passages retained | {np.mean(retained_counts):.1f} |")
    lines.append(f"| Empty expansions | {empty_expansions} |")

    return "\n".join(lines)


def generate_json(results: List[SampleResult], timestamp: str, model: str = LLM_MODEL) -> Dict[str, Any]:
    """Generate structured JSON for further analysis."""
    return {
        "timestamp": timestamp,
        "model": model,
        "sample_count": len(results),
        "summary": {
            "avg_word_count": float(np.mean([len(r.turn2_output.query_expansion.split()) for r in results])),
            "avg_entity_count": float(np.mean([len(r.turn2_output.entities) for r in results])),
            "avg_retained_count": float(np.mean([len(r.turn2_output.retained_ids) for r in results])),
            "empty_expansions": sum(1 for r in results if not r.turn2_output.query_expansion.strip())
        },
        "samples": [
            {
                "sample_num": r.sample_num,
                "current_message": r.current_message,
                "turn1_query_expansion": r.turn1_query_expansion,
                "turn1_memories_count": len(r.turn1_memories),
                "turn2_query_expansion": r.turn2_output.query_expansion,
                "turn2_word_count": len(r.turn2_output.query_expansion.split()),
                "turn2_entities": r.turn2_output.entities,
                "turn2_retained_count": len(r.turn2_output.retained_ids),
                "turn2_raw": r.turn2_output.raw_response,
                "assembled_user_prompt": r.assembled_user_prompt,
            }
            for r in results
        ]
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results: List[SampleResult], args: argparse.Namespace) -> None:
    """Write timestamped JSON and markdown to data/tuning_test_results/subcortical/."""
    now = utc_now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    # Structure: data/tuning_test_results/subcortical/YYYY-MM-DD/{json,md}/
    base_dir = Path("data") / "tuning_test_results" / "subcortical" / date_str
    json_dir = base_dir / "json"
    md_dir = base_dir / "md"
    json_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: time + model slug, with collision avoidance
    model_slug = args.model.split("/")[-1]
    version = f"{now.strftime('%H%M%S')}_{model_slug}"

    json_path = json_dir / f"subcortical_{version}.json"
    md_path = md_dir / f"subcortical_{version}.md"

    counter = 2
    while json_path.exists() or md_path.exists():
        suffixed = f"{version}_{counter}"
        json_path = json_dir / f"subcortical_{suffixed}.json"
        md_path = md_dir / f"subcortical_{suffixed}.md"
        counter += 1

    # JSON data
    json_content = generate_json(results, timestamp, model=args.model)
    json_path.write_text(json.dumps(json_content, indent=2, default=str))
    print(f"\nJSON data: {json_path}")

    # Markdown report
    md_content = generate_markdown(results, timestamp, model=args.model)
    md_path.write_text(md_content)
    print(f"Markdown report: {md_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Subcortical Layer Tuning Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/tuning_harnesses/subcortical_tuning_test.py --seed 1
  python scripts/tuning_harnesses/subcortical_tuning_test.py --seed 3 --model openai/gpt-oss-20b
  python scripts/tuning_harnesses/subcortical_tuning_test.py --seed 1 --save
        """,
    )
    parser.add_argument("--model", type=str, default=LLM_MODEL, help=f"Model ID (default: {LLM_MODEL})")
    parser.add_argument("--endpoint", type=str, default=LLM_ENDPOINT, help=f"API endpoint (default: {LLM_ENDPOINT})")
    parser.add_argument("--api-key-name", type=str, default=LLM_API_KEY_NAME, help=f"Vault key name (default: {LLM_API_KEY_NAME})")
    parser.add_argument("--api-key", type=str, default=None, help="Direct API key (bypasses Vault, for ad-hoc runs)")
    parser.add_argument("--nothink", action="store_true", help="Append /nothink to user template (disables Qwen3 thinking mode)")
    parser.add_argument("--system-prompt", type=str, default=None, help="Path to system prompt file (default: config/prompts/subcortical_system.txt)")
    parser.add_argument("--user-template", type=str, default=None, help="Path to user template file (default: config/prompts/subcortical_user.txt)")
    parser.add_argument("--context-pairs", type=int, default=6, help="Number of conversation turn pairs to include (default: 6, use 3 for trim3 variant)")
    parser.add_argument("--conversation-decay", action="store_true", help="Use conversation decay: recent turns verbose, older turns as keywords. Requires conv_decay variant user template with {older_context}/{recent_turns} placeholders.")
    parser.add_argument("--truncate-passages", type=int, default=0, help="Truncate passage text to first N words (0 = no truncation). E.g. --truncate-passages 15")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for deterministic segment selection (default: 1)")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to run (default: 20)")
    parser.add_argument("--save", action="store_true", help="Save results to data/tuning_test_results/")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-sample terminal output (use with --save)")
    args = parser.parse_args()

    print("=" * 70)
    print("Subcortical Layer Tuning Test")
    print("=" * 70)
    print(f"Model:  {args.model}")
    print(f"Seed:   {args.seed}")
    print()

    # Setup
    print("Setting up user context...")
    user_id = setup_user_context()
    print(f"  User ID: {user_id[:8]}...")

    print("Initializing services...")
    services = setup_services()

    print("Loading prompts from files...")
    system_prompt, user_template = load_prompts(args.system_prompt, args.user_template)
    if args.system_prompt:
        print(f"  System prompt: {args.system_prompt}")
    if args.user_template:
        print(f"  User template: {args.user_template}")
    if args.context_pairs != 6:
        print(f"  Context pairs: {args.context_pairs} (default: 6)")
    if args.conversation_decay:
        print(f"  Conversation decay: ON (recent verbose, older as keywords)")
    if args.truncate_passages > 0:
        print(f"  Passage truncation: {args.truncate_passages} words")
    if args.nothink:
        user_template = user_template.rstrip() + " /nothink"
        print("  /nothink appended to user template")

    # Get samples (seeded for reproducibility)
    random.seed(args.seed)
    print(f"Fetching random message segments (seed={args.seed})...")
    segments = get_random_segments(services["session_manager"], user_id, limit=args.limit)
    print(f"  Found {len(segments)} valid 7-message segments")

    if len(segments) < args.limit:
        print(f"  WARNING: Only {len(segments)} segments available (requested {args.limit})")

    # Run tests
    quiet = args.quiet
    results = []
    for i, messages in enumerate(segments, 1):
        if not quiet:
            print(f"\n{'='*70}")
            print(f"Sample {i}/{len(segments)}")
            print(f"{'='*70}")
        else:
            print(f"  [{i}/{len(segments)}]", end=" ", flush=True)

        result = run_test(
            sample_num=i,
            messages=messages,
            services=services,
            system_prompt=system_prompt,
            user_template=user_template,
            model=args.model,
            endpoint=args.endpoint,
            api_key_name=args.api_key_name,
            api_key_direct=args.api_key,
            quiet=quiet,
            context_pairs=args.context_pairs,
            conversation_decay=args.conversation_decay,
            truncate_passages=args.truncate_passages,
        )
        results.append(result)

        if not quiet:
            # Show the actual output for human evaluation
            print(f"\n  Current message: \"{result.current_message[:80]}{'...' if len(result.current_message) > 80 else ''}\"")
            print(f"\n  --- QUERY EXPANSION ---")
            print(f"  {result.turn2_output.query_expansion}")
            print(f"\n  --- ENTITIES ({len(result.turn2_output.entities)}) ---")
            if result.turn2_output.entities:
                for ent in result.turn2_output.entities:
                    print(f"    • {ent}")
            else:
                print(f"    (none)")
            print(f"\n  --- RETAINED PASSAGES ({len(result.turn2_output.retained_ids)}) ---")
            if result.turn1_memories:
                for mem in result.turn1_memories:
                    mem_short_id = format_memory_id(mem.id).split('_')[1]
                    retained = mem_short_id in result.turn2_output.retained_ids
                    marker = "✓" if retained else "✗"
                    print(f"    [{marker}] {mem.text[:70]}{'...' if len(mem.text) > 70 else ''}")

    if quiet:
        print()  # newline after progress dots

    # Save results if requested
    if args.save:
        save_results(results, args)

    # Final summary (always print — it's compact)
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    word_counts = [len(r.turn2_output.query_expansion.split()) for r in results]
    entity_counts = [len(r.turn2_output.entities) for r in results]

    print(f"\nQuery expansion word counts:")
    print(f"  avg={np.mean(word_counts):.1f}, min={min(word_counts)}, max={max(word_counts)}")

    print(f"\nEntity extraction:")
    print(f"  avg={np.mean(entity_counts):.1f}, min={min(entity_counts)}, max={max(entity_counts)}")

    empty_expansions = sum(1 for r in results if not r.turn2_output.query_expansion.strip())
    print(f"\nEmpty expansions: {empty_expansions}/{len(results)}")

    print(f"\nLLM calls made: {len(results) * 2} (2 per sample)")
    print(f"Samples processed: {len(results)}")


if __name__ == "__main__":
    main()
