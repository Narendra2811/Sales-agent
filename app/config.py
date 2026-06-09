from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from functools import lru_cache


class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key.")
    DATABASE_URL: str = Field(
        default="sqlite:///./sales_agent.db",
        description="SQLAlchemy database URL.",
    )
    CATALOG_PATH: str = Field(
        default="./catalog.json",
        description="Path to the product catalog JSON file.",
    )
    CHROMA_DB_PATH: str = Field(
        default="./chroma_db",
        description="Directory where ChromaDB persists vector embeddings on disk.",
    )
    SHORT_TERM_MESSAGE_LIMIT: int = Field(
        default=10,
        description="How many recent messages to include in each request.",
    )
    SUMMARIZATION_THRESHOLD: int = Field(
        default=20,
        description="Message count threshold for automatic summarization.",
    )
    LLM_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model to use for the agent and evaluation calls.",
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model.",
    )
    EVAL_CONFIDENCE_THRESHOLD: float = Field(
        default=0.7,
        description="Flag responses when confidence is below this threshold.",
    )
    RRF_K_CONSTANT: int = Field(
        default=60,
        description="Reciprocal Rank Fusion k constant.",
    )
    TOP_K_SEARCH_RESULTS: int = Field(
        default=3,
        description="Number of catalog search results to return.",
    )

    PORT: int = Field(
        default=8000,
        description="Port the server listens on.",
    )

    @model_validator(mode="after")
    def fix_postgres_url(self) -> "Settings":
        if self.DATABASE_URL.startswith("postgres://"):
            self.DATABASE_URL = self.DATABASE_URL.replace("postgres://", "postgresql://", 1)
        return self

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# singleton instance
@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
