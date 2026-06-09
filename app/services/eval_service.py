import json
import logging
from typing import Optional

from langchain.chat_models.base import init_chat_model
from langchain_core.messages import HumanMessage

from app.config import settings

logger = logging.getLogger(__name__)

_eval_llm = init_chat_model(
    settings.LLM_MODEL,
    openai_api_key=settings.OPENAI_API_KEY,
    max_tokens=512,
    temperature=0,
)


def run_eval(
    user_message: str,
    agent_response: str,
    memory_context: str,
    catalog_context: str,
) -> dict:
    """
    Scores the agent's response on three dimensions via a second the model call.



    The three scores (0.0 to 1.0):
      - groundedness: Is the response backed by actual catalog data?
      - relevance:    Does it answer the user's question?
      - confidence:   Overall quality (triggers flagging if < threshold)

    Returns:
        dict with keys: groundedness, relevance, confidence, flagged, reasoning
    """
    prompt = f"""You are a quality evaluator for a B2B SaaS sales assistant.

Evaluate the assistant response below. Return ONLY a valid JSON object — no explanation, no markdown.

USER QUESTION:
{user_message}

ASSISTANT RESPONSE:
{agent_response}

CATALOG INFORMATION AVAILABLE TO ASSISTANT:
{catalog_context or "Not retrieved in this turn."}

PRIOR CONVERSATION CONTEXT:
{memory_context or "No prior context."}

Score on these three dimensions (each 0.0 to 1.0):

groundedness: Does every factual claim in the response come directly from the catalog?
  1.0 = 100% catalog-sourced. 0.5 = mix of catalog + plausible guesses. 0.0 = made up.

relevance: Does the response actually answer the user's specific question?
  1.0 = directly and completely answers it. 0.5 = partially answers. 0.0 = off-topic.

confidence: Overall quality. Would a sales manager approve this response?
  1.0 = excellent. 0.7 = acceptable. below 0.7 = needs human review.

Return exactly this JSON:
{{
  "groundedness": <float 0.0-1.0>,
  "relevance": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "flagged": <true if confidence < 0.7, else false>,
  "reasoning": "<one sentence explaining the scores>"
}}"""

    try:
        response = _eval_llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        result.setdefault("groundedness", 0.5)
        result.setdefault("relevance", 0.5)
        result.setdefault("confidence", 0.5)
        result["flagged"] = (
            result.get("confidence", 0.5) < settings.EVAL_CONFIDENCE_THRESHOLD
        )
        result.setdefault("reasoning", "Evaluation completed.")

        for key in ("groundedness", "relevance", "confidence"):
            result[key] = max(0.0, min(1.0, float(result[key])))

        logger.debug(
            f"Eval scores — ground={result['groundedness']:.2f} "
            f"rel={result['relevance']:.2f} conf={result['confidence']:.2f} "
            f"flagged={result['flagged']}"
        )
        return result

    except Exception as e:
        logger.error(f"Eval scoring failed: {e}. Returning safe fallback.")
        return {
            "groundedness": 0.5,
            "relevance": 0.5,
            "confidence": 0.5,
            "flagged": True,
            "reasoning": f"Eval service encountered an error: {str(e)[:100]}",
        }


def extract_facts(
    user_message: str,
    agent_response: str,
) -> list[dict]:
    """
    Extracts new long-term facts about the user from one conversation turn.

    Called after every agent response. The LLM scans the exchange and
    identifies any concrete facts the user revealed about themselves.

    Examples of extractable facts:
      User says "We have 50 employees" -> {"key": "team_size", "value": "50"}
      User says "our CTO will decide"  -> {"key": "decision_maker", "value": "CTO"}
      User says "we're on Starter now" -> {"key": "current_plan", "value": "Starter"}

    Returns:
        List of {"key": str, "value": str} dicts. Empty list if no new facts found.
    """
    prompt = f"""You are a CRM fact extractor for a B2B SaaS sales assistant.

Extract NEW facts the user revealed about themselves from this conversation turn.
Only extract clearly stated facts — never infer or guess.

USER MESSAGE: {user_message}
ASSISTANT RESPONSE: {agent_response}

Return a JSON array. Return [] if no clear new facts were stated.

Common fact keys to look for:
  team_size, company_name, current_plan, budget, pain_point,
  decision_maker, timeline, industry, location, needs_feature,
  num_locations, compliance_requirement

Example return:
[
  {{"key": "team_size", "value": "50"}},
  {{"key": "pain_point", "value": "needs SSO for enterprise clients"}}
]

Return ONLY the JSON array. No markdown. No explanation."""

    try:
        response = _eval_llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        facts = json.loads(raw)

        validated = []
        for fact in facts:
            if isinstance(fact, dict) and "key" in fact and "value" in fact:
                key = str(fact["key"]).strip()
                value = str(fact["value"]).strip()
                if key and value:
                    validated.append({"key": key, "value": value})

        logger.debug(f"Extracted {len(validated)} facts from conversation turn")
        return validated

    except Exception as e:
        logger.warning(f"Fact extraction failed (non-critical): {e}")
        return []  # Non-critical — just return empty, don't crash the request


def generate_summary(messages: list[dict]) -> str:
    """
    Compresses a list of messages into a concise summary paragraph.

    Called when message count exceeds SUMMARIZATION_THRESHOLD.
    The summary replaces the raw older messages in the context window,
    keeping token usage bounded while preserving key information.

    Args:
        messages: List of {"role": str, "content": str} dicts

    Returns:
        A paragraph summarizing the key points of the conversation.
    """
    if not messages:
        return ""

    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    prompt = f"""Summarize the following B2B SaaS sales conversation into a concise paragraph.

Focus on:
- What the user asked about (pricing, features, plans)
- Key facts the user revealed (company size, budget, needs)
- What was already explained to the user
- Any decisions or preferences expressed

Conversation:
{transcript}

Write a 2-4 sentence summary. Be specific. Include numbers and plan names if mentioned."""

    try:
        response = _eval_llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        logger.info(f"Generated summary for {len(messages)} messages")
        return summary
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        lines = [f"{m['role']}: {m['content'][:100]}" for m in messages[-5:]]
        return "Recent conversation: " + " | ".join(lines)
