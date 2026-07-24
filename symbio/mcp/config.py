"""Configuration for the local brain MCP server."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from symbio.constants import PROJECT_DIR


class Settings(BaseSettings):
    """Env-driven settings."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local (Ollama) brain
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_api_key: str | None = Field(default=None, alias="OLLAMA_API_KEY")
    local_model: str = Field(default="llama3.2", alias="LOCAL_MODEL")
    local_temperature: float = Field(default=0.2, alias="LOCAL_TEMPERATURE")
    local_max_tokens: int = Field(default=2048, alias="LOCAL_MAX_TOKENS")
    local_timeout: float = Field(default=120.0, alias="LOCAL_TIMEOUT")

    # Frontier fallback
    frontier_provider: str = Field(default="anthropic", alias="FRONTIER_PROVIDER")
    frontier_model: str = Field(default="claude-sonnet-5-20251001", alias="FRONTIER_MODEL")
    frontier_api_key: str | None = Field(default=None, alias="FRONTIER_API_KEY")
    frontier_temperature: float = Field(default=0.2, alias="FRONTIER_TEMPERATURE")
    frontier_max_tokens: int = Field(default=4096, alias="FRONTIER_MAX_TOKENS")
    frontier_timeout: float = Field(default=120.0, alias="FRONTIER_TIMEOUT")

    # Learning loop
    memory_db_path: Path = Field(default=Path("memory.db"), alias="MEMORY_DB_PATH")
    miss_threshold: int = Field(default=5, alias="MISS_THRESHOLD")
    require_local_first: bool = Field(default=True, alias="REQUIRE_LOCAL_FIRST")
    auto_finetune: bool = Field(default=False, alias="AUTO_FINETUNE")

    @property
    def memory_db(self) -> Path:
        return self.memory_db_path.expanduser().resolve()


settings = Settings()
