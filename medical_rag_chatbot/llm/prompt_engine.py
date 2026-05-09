"""
prompt_engine.py
----------------
Context injection pipeline + prompt engineering for the Medical RAG Chatbot.

Responsibilities:
  1. Format retrieved chunks into a structured context block
  2. Detect query intent (drug interaction, symptom, general Q&A)
  3. Build the system prompt with appropriate medical safety guardrails
  4. Detect and flag drug mentions
"""

import re
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────
_INTERACTION_PATTERNS = re.compile(
    r"\b(interact(?:ion|s)?|combin(?:e|ed|ing)|together|mix(?:ing)?|"
    r"take\s+with|safe\s+with|can\s+I\s+take|should\s+I\s+take|"
    r"drug\s+interaction|side[\s-]effect)\b",
    re.IGNORECASE,
)

_DRUG_NAME_PATTERNS = re.compile(
    r"\b(metformin|lisinopril|atorvastatin|warfarin|ibuprofen|aspirin|"
    r"amoxicillin|metoprolol|amlodipine|omeprazole|simvastatin|losartan|"
    r"gabapentin|sertraline|escitalopram|fluoxetine|clopidogrel|insulin|"
    r"levothyroxine|albuterol|prednisone|hydrochlorothiazide|furosemide|"
    r"methotrexate|clarithromycin|azithromycin|ciprofloxacin|gemfibrozil|"
    r"digoxin|lithium|amiodarone|fluconazole|cimetidine)\b",
    re.IGNORECASE,
)

_SYMPTOM_PATTERNS = re.compile(
    r"\b(symptom|symptoms|sign|signs|pain|ache|fever|fatigue|nausea|"
    r"vomiting|dizziness|shortness\s+of\s+breath|chest\s+pain|headache|"
    r"rash|swelling|bleeding|weakness|confusion)\b",
    re.IGNORECASE,
)


class QueryIntent(NamedTuple):
    is_drug_interaction: bool
    is_symptom_query: bool
    drugs_mentioned: list[str]


def detect_intent(query: str) -> QueryIntent:
    is_interaction = bool(_INTERACTION_PATTERNS.search(query))
    is_symptom     = bool(_SYMPTOM_PATTERNS.search(query))
    drugs          = list({m.group(0).lower() for m in _DRUG_NAME_PATTERNS.finditer(query)})
    return QueryIntent(is_interaction, is_symptom, drugs)


# ─────────────────────────────────────────────────────────────────────────────
# Severity parsing
# ─────────────────────────────────────────────────────────────────────────────
_SEVERITY_MAP = {
    "contraindicated": "🔴 Contraindicated",
    "major":           "🔴 Major",
    "severe":          "🔴 Severe",
    "serious":         "🟠 Serious",
    "avoid":           "🟠 Avoid",
    "moderate":        "🟡 Moderate",
    "caution":         "🟡 Caution",
    "minor":           "🟢 Minor",
}

_SEVERITY_RE = re.compile(
    r"\b(contraindicated|major|severe|serious|avoid|moderate|caution|minor)\b",
    re.IGNORECASE,
)


def extract_severity_label(text: str) -> str:
    m = _SEVERITY_RE.search(text)
    if m:
        return _SEVERITY_MAP.get(m.group(1).lower(), "⚪ Unknown")
    return "⚪ Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Context formatting
# ─────────────────────────────────────────────────────────────────────────────
def format_context(chunks: list[dict], max_chunks: int = 5) -> str:
    """
    Format retrieved chunks into a readable context block for the LLM.
    Each chunk gets a numbered header showing source and category.
    """
    if not chunks:
        return "No relevant context found in the knowledge base."

    parts = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        source   = chunk.get("source", "unknown").upper()
        category = chunk.get("category", "")
        score    = chunk.get("score", 0.0)
        text     = chunk.get("text", "").strip()

        header = f"[{i}] Source: {source}"
        if category:
            header += f" | Topic: {category}"
        header += f" | Relevance: {score:.2f}"

        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────
_BASE_SYSTEM = """You are MedRAG, a medical information assistant powered by MedQuAD and DrugBank databases.

CRITICAL RULES:
1. Answer ONLY using the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say: "I don't have enough information in my database to answer this reliably."
3. NEVER invent drug names, dosages, interactions, or clinical guidelines.
4. Always recommend consulting a licensed healthcare professional or pharmacist.
5. For emergencies, direct the user to call emergency services immediately.

RESPONSE FORMAT:
- Be clear and concise. Use bullet points for lists.
- When citing drug interactions, clearly state the severity level if available.
- End responses with a brief disclaimer when appropriate.
- Do not repeat the context verbatim — synthesize it.

SAFETY:
- Never provide specific dosing advice without noting it requires professional verification.
- Flag any life-threatening interactions prominently."""

_INTERACTION_ADDENDUM = """
DRUG INTERACTION MODE:
- Focus on safety-critical information: severity, mechanism, and clinical management.
- Explicitly state when an interaction is contraindicated.
- List all drugs involved in each interaction clearly.
- If interaction data is incomplete, say so."""

_SYMPTOM_ADDENDUM = """
SYMPTOM QUERY MODE:
- Describe possible conditions associated with these symptoms from the context.
- Do NOT diagnose the user.
- Emphasize that symptoms require professional evaluation."""


def build_system_prompt(intent: QueryIntent) -> str:
    prompt = _BASE_SYSTEM
    if intent.is_drug_interaction:
        prompt += _INTERACTION_ADDENDUM
    if intent.is_symptom_query:
        prompt += _SYMPTOM_ADDENDUM
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_user_message_with_context(query: str, context: str) -> str:
    """Inject the context into the user's message."""
    return (
        f"KNOWLEDGE BASE CONTEXT:\n"
        f"{'='*60}\n"
        f"{context}\n"
        f"{'='*60}\n\n"
        f"USER QUESTION: {query}"
    )


def prepare_llm_inputs(
    query: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Full pipeline: detect intent → format context → build prompt + messages.

    Returns:
        (system_prompt, messages_list)
        messages_list is ready to pass directly to any LLMClient.
    """
    intent  = detect_intent(query)
    context = format_context(chunks)
    system  = build_system_prompt(intent)

    user_content = build_user_message_with_context(query, context)

    # Build message list (preserve chat history for multi-turn)
    messages: list[dict] = []
    if chat_history:
        # Include previous turns but NOT the context-injected ones —
        # just include the clean display versions so history stays readable.
        for turn in chat_history[-6:]:  # last 3 turns (user + assistant pairs)
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Append current user message WITH context
    messages.append({"role": "user", "content": user_content})

    return system, messages
