"""Typed application settings, loaded from ``config.yaml`` plus environment variables.

Precedence (highest first): ``RAG_*`` environment variables, then ``config.yaml``,
then the defaults below. Secrets such as API keys are never stored in the YAML
file; providers name the environment variable that holds their key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.yaml"

RetrievalMode = Literal["hybrid", "dense", "sparse"]


class AppConfig(BaseModel):
    name: str = "rag-search-engine"
    log_level: str = "INFO"
    log_dir: Path = Path("logs")


class PathsConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    model_cache_dir: Path = Path("data/models")
    eval_dir: Path = Path("eval")


class VectorDBConfig(BaseModel):
    url: str = ""
    api_key: str = ""
    path: Path = Path("data/qdrant")
    text_collection: str = "docs"
    image_collection: str = "diagrams"
    upsert_batch_size: int = Field(default=128, gt=0)


class EmbeddingsConfig(BaseModel):
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    image_model: str = "Qdrant/clip-ViT-B-32-vision"
    image_text_model: str = "Qdrant/clip-ViT-B-32-text"
    batch_size: int = Field(default=64, gt=0)


class ChunkingConfig(BaseModel):
    max_chars: int = Field(default=1500, gt=100)
    min_chars: int = Field(default=300, ge=0)

    @model_validator(mode="after")
    def _check_sizes(self) -> ChunkingConfig:
        if self.min_chars >= self.max_chars:
            raise ValueError("chunking.min_chars must be smaller than max_chars")
        return self


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = "hybrid"
    prefetch_limit: int = Field(default=50, gt=0)
    candidates: int = Field(default=30, gt=0)
    top_k: int = Field(default=6, gt=0)
    rerank: RerankConfig = RerankConfig()


class AgentConfig(BaseModel):
    planning: bool = True
    max_sub_queries: int = Field(default=3, ge=1)
    self_correction: bool = True
    max_retrieval_rounds: int = Field(default=2, ge=1)


class GenerationConfig(BaseModel):
    max_context_chunks: int = Field(default=10, gt=0)
    max_tokens: int = Field(default=1500, gt=0)
    max_diagrams: int = Field(default=3, ge=0)


class ProviderConfig(BaseModel):
    """An OpenAI-compatible chat completion endpoint."""

    name: str
    base_url: str
    api_key_env: str
    model: str
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


class LLMConfig(BaseModel):
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 2
    providers: list[ProviderConfig] = Field(default_factory=list)
    # Small, fast models for the planner and grader (short JSON outputs). They
    # have their own rate-limit budget, leaving `providers` for answers.
    # Empty: planner and grader use `providers`.
    fast_providers: list[ProviderConfig] = Field(default_factory=list)
    judge: ProviderConfig | None = None


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)


class UrlRule(BaseModel):
    """Maps a source-relative file path to the document's public URL."""

    base: str = ""
    strip_prefix: str = ""
    replace: dict[str, str] = Field(default_factory=dict)
    # Use this front-matter field as the path when present (e.g. MDN "slug").
    front_matter_key: str = ""

    def build(self, relative_path: str, meta: dict[str, Any] | None = None) -> str:
        if self.front_matter_key and meta and meta.get(self.front_matter_key):
            return self.base + str(meta[self.front_matter_key]).strip("/")
        path = relative_path.removeprefix(self.strip_prefix)
        for old, new in self.replace.items():
            if path.endswith(old):
                path = path[: -len(old)] + new
                break
        return self.base + path


class SourceConfig(BaseModel):
    name: str
    kind: Literal["git", "http_archive", "local"]
    location: str
    format: Literal["markdown", "html", "auto"] = "auto"  # auto: by file extension
    ref: str | None = None
    sparse_paths: list[str] = Field(default_factory=list)
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=list)
    url: UrlRule = UrlRule()
    asset_base: str = ""
    images: bool = False
    download_images: bool = False  # fetch remote diagrams (cached in the source dir)


class DomainConfig(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    sources: list[SourceConfig]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_nested_delimiter="__", extra="ignore"
    )

    app: AppConfig = AppConfig()
    paths: PathsConfig = PathsConfig()
    vectordb: VectorDBConfig = VectorDBConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    agent: AgentConfig = AgentConfig()
    generation: GenerationConfig = GenerationConfig()
    llm: LLMConfig = LLMConfig()
    api: ApiConfig = ApiConfig()
    domains: list[DomainConfig] = Field(default_factory=list)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment variables override the YAML values passed as init kwargs.
        return (env_settings, init_settings)

    @field_validator("domains")
    @classmethod
    def _unique_domains(cls, domains: list[DomainConfig]) -> list[DomainConfig]:
        names = [d.name for d in domains]
        if len(names) != len(set(names)):
            raise ValueError(f"domain names must be unique: {names}")
        return domains

    @property
    def domain_names(self) -> list[str]:
        return [d.name for d in self.domains]

    def domain(self, name: str) -> DomainConfig:
        for domain in self.domains:
            if domain.name == name:
                return domain
        raise KeyError(f"unknown domain {name!r}; configured: {self.domain_names}")

    def resolve_paths(self, root: Path) -> Settings:
        """Make every relative path absolute, relative to ``root``."""

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (root / path).resolve()

        self.app.log_dir = resolve(self.app.log_dir)
        self.paths.raw_data_dir = resolve(self.paths.raw_data_dir)
        self.paths.model_cache_dir = resolve(self.paths.model_cache_dir)
        self.paths.eval_dir = resolve(self.paths.eval_dir)
        self.vectordb.path = resolve(self.vectordb.path)
        return self


def load_settings(config_file: Path | None = None) -> Settings:
    """Load settings from ``config_file`` (default: ``$RAG_CONFIG`` or ``config.yaml``)."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = config_file or Path(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG_FILE))
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings(**data)
    # Conventional names for the vector DB connection, as used by Qdrant tooling.
    settings.vectordb.url = os.environ.get("QDRANT_URL", settings.vectordb.url)
    settings.vectordb.api_key = os.environ.get(
        "QDRANT_API_KEY", settings.vectordb.api_key
    )
    return settings.resolve_paths(path.parent.resolve())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return load_settings()
