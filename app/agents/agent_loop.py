import json
import logging
import uuid
from typing import Annotated, Optional

from langchain.chat_models.base import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import (
    add_messages,
)  # Reducer: appends instead of replacing
from langgraph.prebuilt import ToolNode  # Auto-executes tool calls from AIMessage
from typing_extensions import TypedDict

from app.config import settings
from app.db.database import SessionLocal
from app.db.models import Flag
from app.memory.sqlite_backend import SQLiteMemoryBackend
from app.tools.search_catalog import get_catalog_searcher
from app import services  # Lazy import to avoid circular deps

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    The shared state
    """

    user_id: str  # Who is chatting
    session_id: str  # UUID for this request , new UUID per POST call)
    user_message: str  # The user's latest message

    memory_context: str

    user_facts: str

    messages: Annotated[list, add_messages]

    catalog_context: str

    tools_called: list[str]  # Names of tools that were called
    response: str  # The agent's final text response
    eval_result: dict  # Self-evaluation scores


def _extract_text(content) -> str:
    """
    extract the string text from a LangChain message content field.

    Examples:
        content = "Hello!"                         -> "Hello!"
        content = [{"type": "text", "text": "Hi"}] -> "Hi"
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return " ".join(texts).strip()
    return str(content)


def create_tools(user_id: str, session_id: str) -> list:
    """
    Creates all LangChain tools for this request, with user context baked in.

    Here we use CLOSURES INSTEAD OF MODULE-LEVEL TOOLS

    Args:
        user_id:    The user making this request
        session_id: The session UUID for this request

    Returns:
        List of 3 LangChain tool objects ready to be bound to the LLM.
    """

    catalog_searcher = get_catalog_searcher()

    @tool
    def search_catalog(query: str) -> str:
        """
        Search the SaaSify product catalog for pricing, features, and plan information.
        Use this tool whenever the user asks about plans, pricing, features, limits,
        or any product-specific information.

        Args:
            query: What to search for. Be specific. Examples:
                   "Enterprise plan features", "$499 plan", "SSO authentication",
                   "how many users on Growth plan", "audit logs"
        """
        logger.info(f"Tool called: search_catalog(query='{query[:80]}')")
        result = catalog_searcher.search(query)
        return result

    @tool
    def get_user_memory(context_hint: str = "") -> str:
        """
        Retrieve stored facts and conversation history about the current user.
        Use this tool at the start of complex conversations or when you need to
        recall what the user has told us in previous sessions.

        Args:
            context_hint: Optional description of what you're looking for.
                          e.g., "team size and budget" or "plan preferences"
        """
        logger.info(f"Tool called: get_user_memory(hint='{context_hint[:50]}')")
        with SessionLocal() as db:
            backend = SQLiteMemoryBackend(db)
            return backend.get_user_context_string(user_id)

    @tool
    def flag_for_human(reason: str, confidence_score: float = 0.4) -> str:
        """
        Escalate this conversation to a human sales representative.
        Use this tool when:
        - You cannot find relevant information in the catalog
        - The user asks about custom pricing or negotiation
        - The user requests a demo or a call
        - You are not confident your answer is accurate

        Args:
            reason:           Why you are escalating (be specific)
            confidence_score: Your confidence level (0.0 to 1.0). Default 0.4.
        """
        logger.info(
            f"Tool called: flag_for_human("
            f"reason='{reason[:80]}', confidence={confidence_score})"
        )
        try:
            with SessionLocal() as db:
                flag = Flag(
                    user_id=user_id,
                    session_id=session_id,
                    reason=reason,
                    confidence=confidence_score,
                    resolved=False,
                )
                db.add(flag)
                db.commit()
            return (
                f"Conversation flagged for human review. "
                f"A sales representative will follow up. "
                f"Reason logged: {reason}"
            )
        except Exception as e:
            logger.error(f"flag_for_human tool failed: {e}")
            return "Flagging noted, but could not persist to database."

    return [search_catalog, get_user_memory, flag_for_human]


def load_memory_node(state: AgentState) -> dict:
    """
    NODE 1: load_memory
    Reads both memory tiers from the database before calling the LLM.

    SHORT-TERM (memory_context):
      = [Latest summary paragraph] + [Last K raw messages]
      Gives the agent recent conversational context.

    LONG-TERM (user_facts):
      = All extracted key-value facts about this user
      Gives the agent persistent knowledge about this user's situation.

    """
    user_id = state["user_id"]
    logger.info(f"[load_memory] Loading memory for user={user_id}")

    try:
        with SessionLocal() as db:
            backend = SQLiteMemoryBackend(db)

            memory_context = backend.get_memory_context_string(
                user_id, recent_limit=settings.SHORT_TERM_MESSAGE_LIMIT
            )

            user_facts = backend.get_user_facts_string(user_id)

        logger.debug(
            f"[load_memory] Loaded {len(memory_context)} chars context, "
            f"{len(user_facts)} chars facts"
        )
    except Exception as e:
        logger.error(f"[load_memory] Failed to load memory: {e}")
        memory_context = ""
        user_facts = ""

    return {
        "memory_context": memory_context,
        "user_facts": user_facts,
    }


def create_call_llm_node(llm_with_tools):
    """
    Factory that creates the call_llm node with a pre-configured LLM.

    Returns:
        A node function: (AgentState) -> dict
    """

    def call_llm_node(state: AgentState) -> dict:
        """
        NODE 2 (and loops back): call_llm
        Sends the conversation to the model with tools available.

        Builds a system prompt that includes:
          - Agent persona and instructions
          - Long-term user facts (who is this user?)
          - Short-term memory context (what were we just talking about?)

        The LLM either:
          a) Returns a final text response (no tool calls) -> go to save_and_eval
          b) Returns tool call requests -> go to run_tools -> come back here

        This node can execute MULTIPLE TIMES per request (the ReAct loop).
        """
        user_id = state["user_id"]
        session_id = state["session_id"]
        logger.info(f"[call_llm] Calling LLM for user={user_id}")

        user_facts_section = ""
        if state.get("user_facts"):
            user_facts_section = f"""
=== What We Know About This User (Long-Term Memory) ===
{state['user_facts']}
"""

        memory_context_section = ""
        if state.get("memory_context"):
            memory_context_section = f"""
=== Conversation History ===
{state['memory_context']}
"""

        system_prompt = f"""You are an expert sales assistant for SaaSify, a B2B SaaS platform.
Your goal is to help potential customers find the right plan for their needs.

ALWAYS use the search_catalog tool before answering questions about pricing, features, or plans.
ALWAYS use the get_user_memory tool if you need to recall prior context about this user.
NEVER make up pricing or feature information — only use what the catalog tool returns.
If you are unsure or the user wants custom pricing/demo, call the flag_for_human tool.

Current user ID: {user_id}
Current session: {session_id}
{user_facts_section}{memory_context_section}
Tone: Professional, helpful, and consultative. Ask clarifying questions when relevant."""

        messages_to_send = [SystemMessage(content=system_prompt)] + state["messages"]

        try:
            ai_response = llm_with_tools.invoke(messages_to_send)
            logger.debug(
                f"[call_llm] LLM responded. "
                f"tool_calls={len(ai_response.tool_calls) if ai_response.tool_calls else 0}"
            )
        except Exception as e:
            logger.error(f"[call_llm] LLM call failed: {e}")
            ai_response = AIMessage(
                content="I'm sorry, I encountered a technical issue. Please try again.",
                tool_calls=[],
            )

        return {"messages": [ai_response]}

    return call_llm_node


def create_save_and_eval_node(user_id: str, session_id: str):
    """
    Factory that creates the save_and_eval node with user context baked in.
    This is the FINAL node — it runs after the LLM gives its final response.
    """

    def save_and_eval_node(state: AgentState) -> dict:
        """
        NODE 3: save_and_eval (final node)
        After the LLM finishes generating a response, this node:
          1. Extracts the final response text
          2. Collects which tools were called during this request
          3. Collects catalog context (for eval grounding)
          4. Saves user + assistant messages to the DB
          5. Runs fact extraction (background: update long-term memory)
          6. Runs self-evaluation (scores groundedness, relevance, confidence)
          7. Saves eval to DB
          8. Triggers memory summarization if message count > threshold
          9. Closes the session record
         10. Returns final response + eval block
        """
        logger.info(f"[save_and_eval] Finalizing response for user={user_id}")

        messages = state["messages"]

        final_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_ai_message = msg
                break

        response_text = (
            _extract_text(final_ai_message.content) if final_ai_message else ""
        )
        if not response_text:
            response_text = (
                "I'm sorry, I wasn't able to generate a response. Please try again."
            )

        tools_called = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = (
                        tc.get("name", "")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "")
                    )
                    if name and name not in tools_called:
                        tools_called.append(name)

        catalog_context_parts = []
        for msg in messages:
            if hasattr(msg, "name") and msg.name == "search_catalog":
                catalog_context_parts.append(_extract_text(msg.content))
        catalog_context = "\n\n".join(catalog_context_parts)

        try:
            with SessionLocal() as db:
                backend = SQLiteMemoryBackend(db)

                backend.save_message(
                    user_id=user_id,
                    session_id=session_id,
                    role="user",
                    content=state["user_message"],
                )

                backend.save_message(
                    user_id=user_id,
                    session_id=session_id,
                    role="assistant",
                    content=response_text,
                )

                total_messages = backend.get_message_count(user_id)
        except Exception as e:
            logger.error(f"[save_and_eval] Failed to save messages: {e}")
            total_messages = 0

        try:
            from app.services import eval_service

            new_facts = eval_service.extract_facts(
                user_message=state["user_message"],
                agent_response=response_text,
            )
            if new_facts:
                with SessionLocal() as db:
                    backend = SQLiteMemoryBackend(db)
                    for fact in new_facts:
                        backend.save_user_fact(
                            user_id=user_id,
                            fact_key=fact["key"],
                            fact_value=fact["value"],
                            confidence=0.9,
                        )
                logger.info(
                    f"[save_and_eval] Saved {len(new_facts)} new facts for user={user_id}"
                )
        except Exception as e:
            logger.warning(
                f"[save_and_eval] Fact extraction failed (non-critical): {e}"
            )

        try:
            from app.services import eval_service

            eval_result = eval_service.run_eval(
                user_message=state["user_message"],
                agent_response=response_text,
                memory_context=state.get("memory_context", ""),
                catalog_context=catalog_context,
            )
        except Exception as e:
            logger.error(f"[save_and_eval] Eval failed: {e}")
            eval_result = {
                "groundedness": 0.5,
                "relevance": 0.5,
                "confidence": 0.5,
                "flagged": True,
                "reasoning": f"Eval error: {str(e)[:80]}",
            }

        try:
            with SessionLocal() as db:
                backend = SQLiteMemoryBackend(db)
                backend.save_eval(
                    user_id=user_id,
                    session_id=session_id,
                    groundedness=eval_result["groundedness"],
                    relevance=eval_result["relevance"],
                    confidence=eval_result["confidence"],
                    flagged=eval_result["flagged"],
                    reasoning=eval_result.get("reasoning", ""),
                    tools_called=tools_called,
                )
        except Exception as e:
            logger.error(f"[save_and_eval] Failed to save eval: {e}")

        if total_messages > settings.SUMMARIZATION_THRESHOLD:
            _try_summarize(user_id, session_id)

        try:
            with SessionLocal() as db:
                backend = SQLiteMemoryBackend(db)
                backend.close_session(session_id, message_count=total_messages)
        except Exception as e:
            logger.warning(f"[save_and_eval] Failed to close session: {e}")

        logger.info(
            f"[save_and_eval] Done. tools={tools_called} "
            f"conf={eval_result.get('confidence', 0):.2f} "
            f"flagged={eval_result.get('flagged', False)}"
        )

        return {
            "response": response_text,
            "eval_result": eval_result,
            "tools_called": tools_called,
            "catalog_context": catalog_context,
        }

    return save_and_eval_node


def _try_summarize(user_id: str, session_id: str) -> None:
    """
    Helper: attempt to summarize older messages for a user.
    Runs as a best-effort step — errors are logged but don't crash the request.
    """
    try:
        with SessionLocal() as db:
            backend = SQLiteMemoryBackend(db)

            all_messages = backend.get_recent_messages(
                user_id, limit=settings.SUMMARIZATION_THRESHOLD
            )

            if not all_messages:
                return

            from app.services import eval_service

            summary_text = eval_service.generate_summary(all_messages)

            if summary_text:
                backend.save_summary(
                    user_id=user_id,
                    session_id=session_id,
                    summary_text=summary_text,
                    message_count_covered=len(all_messages),
                )
                logger.info(
                    f"Auto-summarized {len(all_messages)} messages for user={user_id}"
                )
    except Exception as e:
        logger.warning(f"Auto-summarization failed (non-critical): {e}")


def should_use_tools(state: AgentState) -> str:
    """
    Conditional edge: decides where to go after call_llm.

    The loop ends when the model gives a final response with no tool calls.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug(
            f"[routing] Tool calls detected: "
            f"{[tc.get('name', '') if isinstance(tc, dict) else tc.name for tc in last_message.tool_calls]}"
        )
        return "tools"

    logger.debug("[routing] No tool calls — routing to save_and_eval")
    return "save_and_eval"


class SalesAgentRunner:
    """
    The main agent class. Builds and runs the LangGraph for each request.

    Usage:
        runner = SalesAgentRunner()          # Create once at startup
        result = runner.run(                 # Call for each request
            user_id="alice",
            session_id="uuid-123",
            user_message="What's Enterprise pricing?"
        )

    Design:
      One SalesAgentRunner instance per application.
      Per request, a new LangGraph is compiled with fresh tools (closures).
      This is lightweight because LangGraph compilation is fast.
    """

    def __init__(self):
        self._base_llm = init_chat_model(
            settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=1024,
            temperature=0.3,  # Slight creativity but mostly deterministic
        )
        logger.info(f"SalesAgentRunner initialized with model={settings.LLM_MODEL}")

    def run(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
    ) -> dict:

        logger.info(
            f"SalesAgentRunner.run() user={user_id} session={session_id[:8]}..."
        )

        tools = create_tools(user_id, session_id)

        llm_with_tools = self._base_llm.bind_tools(tools)

        graph = self._build_graph(tools, llm_with_tools, user_id, session_id)

        initial_state: AgentState = {
            "user_id": user_id,
            "session_id": session_id,
            "user_message": user_message,
            "memory_context": "",  # Filled by load_memory_node
            "user_facts": "",  # Filled by load_memory_node
            "messages": [
                HumanMessage(content=user_message)
            ],  # Starts with user's message
            "catalog_context": "",  # Filled during tool execution
            "tools_called": [],  # Filled by save_and_eval_node
            "response": "",  # Filled by save_and_eval_node
            "eval_result": {},  # Filled by save_and_eval_node
        }

        try:
            final_state = graph.invoke(
                initial_state,
                config={
                    "recursion_limit": 10
                },  # Max 10 node executions (prevents infinite loops)
            )
        except Exception as e:
            logger.error(f"Graph execution failed for user={user_id}: {e}")
            return {
                "response": "I encountered a technical issue. Please try again.",
                "eval": {
                    "groundedness": 0.0,
                    "relevance": 0.0,
                    "confidence": 0.0,
                    "flagged": True,
                    "reasoning": f"System error: {str(e)[:100]}",
                },
                "tools_called": [],
                "session_id": session_id,
            }

        return {
            "response": final_state.get("response", "No response generated."),
            "eval": final_state.get("eval_result", {}),
            "tools_called": final_state.get("tools_called", []),
            "session_id": session_id,
        }

    def _build_graph(
        self,
        tools: list,
        llm_with_tools,
        user_id: str,
        session_id: str,
    ):

        builder = StateGraph(AgentState)

        builder.add_node("load_memory", load_memory_node)
        builder.add_node("call_llm", create_call_llm_node(llm_with_tools))
        builder.add_node("tools", ToolNode(tools))
        builder.add_node(
            "save_and_eval", create_save_and_eval_node(user_id, session_id)
        )

        builder.set_entry_point("load_memory")
        builder.add_edge("load_memory", "call_llm")

        builder.add_conditional_edges(
            "call_llm",
            should_use_tools,
            {
                "tools": "tools",  # Yes tool calls -> execute them
                "save_and_eval": "save_and_eval",  # No tool calls -> finalize
            },
        )

        builder.add_edge("tools", "call_llm")

        builder.add_edge("save_and_eval", END)

        return builder.compile()


_agent_runner: Optional[SalesAgentRunner] = None


def get_agent_runner() -> SalesAgentRunner:
    """Returns the singleton SalesAgentRunner, creating it if needed."""
    global _agent_runner
    if _agent_runner is None:
        _agent_runner = SalesAgentRunner()
    return _agent_runner
