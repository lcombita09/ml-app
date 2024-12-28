"""
This module sets up the logger configuration.

It use Pydantic's BaseSettings for configuration management,
allowing settings to be read from environment variables and a .env file.
"""

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
        )

    log_level: str


def configure_logger(log_level: str) -> None:
    """
    Configure the logging for the application.

    Arg:
        log_level (str): The log level to be set for the logger.

    Return:
        None
    """
    logger.add(
        'logs/logs.log',
        rotation='500 MB',
        retention='10 days',
        level=log_level,
        compression='zip',
        )


configure_logger(log_level=LoggerSettings().log_level)
