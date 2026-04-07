"""Application settings — config loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config object. Values are read from environment variables or .env file.

    Never hardcode secrets. Always instantiate this class to access config:

        settings = Settings()
        key = settings.anthropic_api_key
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- LLM (Anthropic) ---
    anthropic_api_key: str = Field(description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model used by all agents",
    )

    # --- Embeddings (local sentence-transformers, no API key required) ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model for FAISS ingestion",
    )

    # --- LangSmith tracing ---
    langchain_api_key: str = Field(default="", description="LangSmith API key")
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    langchain_project: str = Field(
        default="mosaic-agent",
        description="LangSmith project name",
    )

    # --- SEC EDGAR ---
    sec_user_agent: str = Field(
        default="mosaic-agent research@example.com",
        description=(
            "User-Agent header required by SEC EDGAR. "
            "Format: 'AppName email@domain.com'. "
            "Override via SEC_USER_AGENT env var."
        ),
    )
    edgar_form_types: list[str] = Field(
        default=["10-K", "10-Q", "8-K"],
        description='Filing types to ingest. Set as JSON: \'["10-K","10-Q"]\'',
    )
    edgar_max_filings: int = Field(
        default=3,
        description="Max filings to fetch per run (across all form types)",
    )

    # --- RAG chunking ---
    chunk_size: int = Field(default=1000, description="Chunk size for text splitter")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    retrieval_k: int = Field(default=6, description="Number of chunks returned by MMR")
    retrieval_fetch_k: int = Field(
        default=20,
        description="Candidate pool size for MMR diversity re-ranking",
    )

    # --- Re-ranker ---
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace cross-encoder model for re-ranking hybrid results",
    )

    # --- FAISS persistence ---
    faiss_index_dir: str = Field(
        default=".faiss_cache",
        description="Directory where FAISS indexes are persisted per ticker",
    )

    # --- Graph ---
    max_research_iterations: int = Field(
        default=3,
        description="Max research→signal loop iterations before giving up",
    )
    confidence_threshold: float = Field(
        default=0.6,
        description="Minimum signal confidence to proceed to backtesting",
    )
