from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    MLFLOW_TRACKING_URI: str = ""
    MLFLOW_EXPERIMENT_NAME: str = ""
    API_SECRET_KEY: str
    API_CORS_ORIGINS: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
