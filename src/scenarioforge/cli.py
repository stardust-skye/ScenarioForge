"""Shared command-line initialization."""

from dotenv import load_dotenv

from scenarioforge.config.settings import PROJECT_ROOT, Settings


def load_settings() -> Settings:
    """Load repository-local environment overrides, then build settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings.from_env()
