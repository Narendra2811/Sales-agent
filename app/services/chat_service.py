
import logging
import uuid

from app.agents.agent_loop import get_agent_runner
from app.db.database import SessionLocal
from app.memory.sqlite_backend import SQLiteMemoryBackend

logger = logging.getLogger(__name__)


def handle_chat(user_id: str, user_message: str) -> dict:
    """
    Handles one complete POST /chat/{user_id} request.

    Steps:
      1. Generate a new session UUID (every request = new session)
      2. Create the session record in the DB
      3. Run the LangGraph agent
      4. Return the structured response

    Args:
        user_id:      From the URL path parameter
        user_message: From the request body

    Returns:
        {
            "response":     str,   # Agent's answer
            "eval":         dict,  # Self-eval scores
            "tools_called": list,  # Tools used
            "session_id":   str,   # For debugging/tracing
        }
    """
    session_id = str(uuid.uuid4())
    logger.info(f"[chat_service] user={user_id} session={session_id[:8]} msg='{user_message[:60]}'")

    try:
        with SessionLocal() as db:
            backend = SQLiteMemoryBackend(db)
            backend.create_session(session_id=session_id, user_id=user_id)
    except Exception as e:
        logger.warning(f"[chat_service] Non-critical: session creation failed: {e}")

    runner = get_agent_runner()
    result = runner.run(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
    )

    return result


def get_user_history(user_id: str) -> dict:
    """
    Fetches full conversation history for GET /chat/{user_id}/history.
    Returns all messages across all sessions in chronological order.
    """
    try:
        with SessionLocal() as db:
            backend = SQLiteMemoryBackend(db)
            messages = backend.get_full_history(user_id)
            return {
                "user_id": user_id,
                "total_messages": len(messages),
                "messages": messages,
            }
    except Exception as e:
        logger.error(f"[chat_service] Failed to get history for user={user_id}: {e}")
        return {"user_id": user_id, "total_messages": 0, "messages": []}


def delete_user_memory(user_id: str) -> dict:
    """
    Deletes all memory for a user (GDPR right to erasure).
    Called by DELETE /chat/{user_id}/memory.
    """
    try:
        with SessionLocal() as db:
            backend = SQLiteMemoryBackend(db)
            deleted = backend.delete_user_memory(user_id)
            return {
                "user_id": user_id,
                "deleted": deleted,
                "message": "All memory data permanently deleted.",
            }
    except Exception as e:
        logger.error(f"[chat_service] Failed to delete memory for user={user_id}: {e}")
        raise


def get_user_evals(user_id: str) -> dict:
    """
    Returns aggregated eval stats for GET /chat/{user_id}/evals.
    Powers the bonus analytics endpoint.
    """
    try:
        with SessionLocal() as db:
            backend = SQLiteMemoryBackend(db)
            stats = backend.get_eval_aggregate(user_id)
            return {"user_id": user_id, **stats}
    except Exception as e:
        logger.error(f"[chat_service] Failed to get evals for user={user_id}: {e}")
        return {"user_id": user_id, "total_responses": 0, "error": str(e)}
