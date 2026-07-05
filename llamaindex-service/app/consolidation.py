"""Consolidation: distill raw turns (L2) into long-term facts (L3).

Two modes, depending on whether the service has its own LLM:
- LLM configured: the service extracts facts itself (endpoint or MCP tool).
- LLM_PROVIDER=none: `consolidate_session` returns the transcript plus these
  extraction instructions, the Hermes-side model does the distillation and
  hands the facts back through `save_memories`.
"""

import json

from qdrant_client import QdrantClient

from app import config, memories, memory_store

EXTRACTION_INSTRUCTIONS = """\
You are a long-term memory curator. From the transcript, extract ONLY facts
that will still be useful in FUTURE conversations:
- durable decisions the user made
- stable user preferences (style, tools, workflow)
- concrete info about the user or their projects (names, roles, deadlines,
  constraints, tech stack)
- commitments that extend beyond this conversation

Do NOT extract (this is the most common failure — be strict):
- general or world knowledge, explanations, tutorials (even if correct)
- descriptions or capabilities of any software/tool/AI assistant,
  INCLUDING yourself and the system you run on
- summaries of advice or analysis that was given during the chat
- meta-commentary about the conversation itself or the memory system's state
- transient tasks that resolve within this conversation

Each fact must be self-contained (understandable without the transcript),
in the same language as the conversation. Return AT MOST {max_facts} facts —
pick only the most durable. When in doubt, LEAVE IT OUT; returning [] is a
perfectly good answer for chit-chat or purely informational sessions.

Return ONLY a JSON array (no prose, no code fences):
[{{"text": "...", "type": "fact|preference|decision|task", "importance": 0.0-1.0}}]"""


MAX_TRANSCRIPT_CHARS = 12000  # keep extraction prompts fast; newest turns win


def transcript_from_points(points: list) -> str:
    lines = [f"{p.payload['role']}: {p.payload['content']}" for p in points]
    transcript = "\n".join(lines)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        kept: list[str] = []
        total = 0
        for line in reversed(lines):  # newest turns carry the decisions
            if total + len(line) > MAX_TRANSCRIPT_CHARS:
                break
            kept.append(line)
            total += len(line)
        transcript = "[...phần đầu hội thoại đã lược bớt...]\n" + "\n".join(reversed(kept))
    return transcript


def _parse_facts(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [f for f in parsed if isinstance(f, dict) and f.get("text")]


def extract_with_llm(llm, transcript: str) -> list[dict]:
    instructions = EXTRACTION_INSTRUCTIONS.format(max_facts=config.CONSOLIDATION_MAX_FACTS)
    prompt = f"{instructions}\n\nTranscript:\n{transcript}"
    facts = _parse_facts(llm.complete(prompt).text)
    # Belt and braces on top of the prompt: importance floor + hard cap.
    facts = [
        f for f in facts
        if float(f.get("importance", 0.5)) >= config.CONSOLIDATION_MIN_IMPORTANCE
    ]
    facts.sort(key=lambda f: float(f.get("importance", 0.5)), reverse=True)
    return facts[: config.CONSOLIDATION_MAX_FACTS]


def consolidate_session(
    client: QdrantClient,
    embed_model,
    llm,
    session_id: str,
    user_id: str = config.USER_ID,
) -> dict:
    points = memory_store.fetch_unconsolidated(client, session_id, user_id)
    if not points:
        return {"status": "nothing_to_consolidate", "facts": []}
    if llm is None:
        raise RuntimeError(
            "No LLM configured (LLM_PROVIDER=none). Use the "
            "`consolidate_session` MCP tool from Hermes instead, or set an "
            "LLM provider in .env."
        )

    facts = extract_with_llm(llm, transcript_from_points(points))
    project_id = points[0].payload.get("project_id") or config.DEFAULT_PROJECT
    saved = memories.save_facts(
        client, embed_model, facts, user_id, session_id, project_id=project_id
    )
    memory_store.mark_consolidated(client, [p.id for p in points])
    return {"status": "ok", "turns_processed": len(points), "project": project_id, "facts": saved}
