from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_stt_model: str = "whisper-large-v3"
    groq_llm_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    database_url: str = "sqlite:///./data/scribe.db"
    chroma_persist_dir: str = "./data/chroma"

    # Retrieval defaults
    rrf_k: int = 60
    retrieval_top_k: int = 5
    half_life_days: dict[str, float | None] = {
        "visit": 365.0,
        "note": 365.0,
        "lab": 180.0,
        "allergy": None,     # No decay
        "diagnosis": None,   # Chronic diagnoses do not decay
    }

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()