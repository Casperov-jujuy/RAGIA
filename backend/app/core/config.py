from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración centralizada de RAGIA leída desde variables de entorno y archivos .env.
    """

    # Configuración de Google Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_LLM_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # Rutas de almacenamiento
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    UPLOAD_DIR: str = "./data/uploads"

    # Servidor y Entorno
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def has_valid_gemini_key(self) -> bool:
        """Determina si existe una API key de Gemini real (no vacía ni placeholder)."""
        key = self.GEMINI_API_KEY.strip()
        return bool(key and not key.startswith("tu_api_key"))


settings = Settings()
