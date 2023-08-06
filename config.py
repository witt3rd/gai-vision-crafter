from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


@lru_cache()
def get_config():
    return Settings(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        SYSTEM_PROMPT_FILE=os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt"),
        JOBS_DIR=os.getenv("JOBS_DIR", "./jobs"),
    )


@dataclass
class Settings:
    OPENAI_API_KEY: str
    SYSTEM_PROMPT_FILE: str
    JOBS_DIR: str
