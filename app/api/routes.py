
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.db.database import verify_db_connection
from app.models.schemas import (
    ChatRequest, ChatResponse, EvalBlock,
    HistoryResponse, MessageRecord,
    MemoryDeleteResponse,
    EvalAggregateResponse, EvalRecord,
    CatalogResponse, HealthResponse,
)
from app.services import chat_service
from app.tools.search_catalog import get_catalog_searcher

logger = logging.getLogger(__name__)

router = APIRouter()



@router.post(
    "/chat/{user_id}",
    response_model=ChatResponse,
    summary="Send a message to the sales assistant",
    description=(
        "Send a message and receive a response with self-evaluation scores. "
        "Memory persists across multiple calls with the same user_id."
    ),
    tags=["Chat"],
)
def chat(user_id: str, body: ChatRequest) -> ChatResponse:
    """
    The main endpoint. One call = one full agent turn.

    What happens behind the scenes:
      1. chat_service generates a session UUID
      2. LangGraph loads the user's memory from the DB
      3. the model calls search_catalog and/or get_user_memory tools as needed
      4. the model generates a response
      5. Eval service scores the response
      6. Facts are extracted and saved to long-term memory
      7. Response + eval block returned

    Returns 500 if the agent encounters an unrecoverable error.
    """
    try:
        result = chat_service.handle_chat(
            user_id=user_id,
            user_message=body.message,
        )

        return ChatResponse(
            response=result["response"],
            eval=EvalBlock(**result["eval"]),
            tools_called=result.get("tools_called", []),
            session_id=result["session_id"],
        )

    except Exception as e:
        logger.error(f"POST /chat/{user_id} failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(e)}"
        )



@router.get(
    "/chat/{user_id}/history",
    response_model=HistoryResponse,
    summary="Get full conversation history",
    description="Returns all messages for this user across all sessions, in chronological order.",
    tags=["Chat"],
)
def get_history(user_id: str) -> HistoryResponse:
    """
    Returns the complete conversation history for a user.
    Useful for the frontend to render past conversations.
    """
    try:
        data = chat_service.get_user_history(user_id)
        messages = [MessageRecord.model_validate(m) for m in data["messages"]]
        return HistoryResponse(
            user_id=user_id,
            total_messages=data["total_messages"],
            messages=messages,
        )
    except Exception as e:
        logger.error(f"GET /chat/{user_id}/history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.delete(
    "/chat/{user_id}/memory",
    response_model=MemoryDeleteResponse,
    summary="Delete all memory for a user (GDPR)",
    description=(
        "Permanently deletes all stored data for this user: "
        "messages, facts, summaries, sessions, evals, and flags. "
        "This action is irreversible."
    ),
    tags=["Chat"],
)
def delete_memory(user_id: str) -> MemoryDeleteResponse:
    """GDPR right-to-erasure endpoint. Permanently deletes all user data."""
    try:
        result = chat_service.delete_user_memory(user_id)
        return MemoryDeleteResponse(
            user_id=user_id,
            deleted=result["deleted"],
            message=result["message"],
        )
    except Exception as e:
        logger.error(f"DELETE /chat/{user_id}/memory failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get(
    "/chat/{user_id}/evals",
    response_model=EvalAggregateResponse,
    summary="Get aggregated eval scores (bonus)",
    description=(
        "Returns quality metrics aggregated across all sessions for a user. "
        "Shows average groundedness, relevance, confidence, and % flagged."
    ),
    tags=["Evals"],
)
def get_evals(user_id: str) -> EvalAggregateResponse:
    """
    Bonus endpoint: aggregated response quality stats per user.
    Can be used to build a quality dashboard for the sales team.
    """
    try:
        data = chat_service.get_user_evals(user_id)
        recent = [EvalRecord(**e) for e in data.get("recent_evals", [])]
        return EvalAggregateResponse(
            user_id=user_id,
            total_responses=data.get("total_responses", 0),
            avg_groundedness=data.get("avg_groundedness", 0.0),
            avg_relevance=data.get("avg_relevance", 0.0),
            avg_confidence=data.get("avg_confidence", 0.0),
            pct_flagged=data.get("pct_flagged", 0.0),
            total_flagged=data.get("total_flagged", 0),
            recent_evals=recent,
        )
    except Exception as e:
        logger.error(f"GET /chat/{user_id}/evals failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get(
    "/catalog",
    response_model=CatalogResponse,
    summary="Get the product catalog",
    description="Returns the full SaaSify product catalog JSON.",
    tags=["Catalog"],
)
def get_catalog() -> CatalogResponse:
    """Returns the raw product catalog that the agent uses for answers."""
    try:
        with open(settings.CATALOG_PATH, "r") as f:
            catalog_data = json.load(f)
        return CatalogResponse(catalog=catalog_data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Catalog file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns the health status of the API and its dependencies.",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """
    Health check endpoint used by Railway and monitoring tools.
    Returns 200 even if some components are degraded (to avoid restart loops).
    """
    db_ok = verify_db_connection()

    catalog_ok = Path(settings.CATALOG_PATH).exists()

    try:
        searcher = get_catalog_searcher()
        vector_ok = searcher.collection.count() > 0
    except Exception:
        vector_ok = False

    overall = "healthy" if (db_ok and catalog_ok and vector_ok) else "degraded"

    return HealthResponse(
        status=overall,
        database="connected" if db_ok else "error",
        catalog="loaded" if catalog_ok else "error",
        vector_store="initialized" if vector_ok else "error",
        version="1.0.0",
    )
