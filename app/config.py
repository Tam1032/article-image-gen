from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (or a .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Services
    comfyui_url: str = "http://comfyui:8188"
    ollama_url: str = "http://ollama:11434"
    redis_url: str = "redis://redis:6379/0"

    # LLM (prompt writer)
    llm_model: str = "qwen3.5:9b"
    llm_think: bool = False
    llm_temperature: float = 0.7
    llm_timeout_s: float = 300

    # Image generation
    default_workflow: str = "flux_txt2img"
    workflows_dir: Path = Path("workflows")
    generation_timeout_s: float = 600
    # On a single 24 GB GPU the LLM and FLUX cannot both stay in VRAM.
    # When true, each stage releases the GPU before the next one starts.
    free_vram_between_stages: bool = True

    # Output
    output_dir: Path = Path("data/outputs")
    public_base_url: str = "http://localhost:8000"
    webp_quality: int = 85
    job_ttl_s: int = 7 * 24 * 3600


@lru_cache
def get_settings() -> Settings:
    return Settings()
