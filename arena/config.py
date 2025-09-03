from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """LLM configuration."""
    model: str = Field(default="gpt-4", description="Model name")
    max_tokens: int = Field(default=4000, description="Maximum tokens")
    temperature: float = Field(default=0.7, description="Temperature for generation")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # api keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # LLM Settings
    default_model: str = "gpt-4o-mini"
    max_tokens: int = 4000
    temperature: float = 0.7

    # logging
    debug: bool = False
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()
