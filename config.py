import os
from dotenv import load_dotenv

load_dotenv()

PROJECTS_PATH: str = os.getenv("SEEK_PROJECTS_PATH", "C:/Projects")
EMBED_MODEL: str = os.getenv("SEEK_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE: int = int(os.getenv("SEEK_CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("SEEK_CHUNK_OVERLAP", "100"))
TOP_K_DEFAULT: int = int(os.getenv("SEEK_TOP_K", "5"))

INCLUDED_PATTERNS: list[str] = [
    "*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java", "*.cpp", "*.c", "*.cs",
]
EXCLUDED_PATTERNS: list[str] = [
    "node_modules/**", ".git/**", "*.min.js", "__pycache__/**",
    ".venv/**", "venv/**", "dist/**", "build/**",
]

def database_url() -> str:
    url = os.getenv("COCOINDEX_DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "COCOINDEX_DATABASE_URL is not set. "
            "Copy .env.example to .env and fill in your Postgres password."
        )
    return url
