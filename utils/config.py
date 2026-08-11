import os
from pathlib import Path

from dotenv import load_dotenv


UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent
ENV_FILE = UTILS_DIR / ".env"

load_dotenv(ENV_FILE)


def project_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


DATA_DIR = project_path(
    os.getenv("RAG_DATA_DIR", "data")
)

CHROMA_DIR = project_path(
    os.getenv("RAG_CHROMA_DIR", "chroma_db")
)

COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION_NAME",
    "knowledge_base",
)

ARK_BASE_URL = os.getenv(
    "ARK_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/plan/v3",
)

ARK_EMBEDDING_MODEL = os.getenv(
    "ARK_EMBEDDING_MODEL",
    "doubao-embedding-vision",
)

CHUNK_SIZE = int(
    os.getenv("RAG_CHUNK_SIZE", "800")
)

CHUNK_OVERLAP = int(
    os.getenv("RAG_CHUNK_OVERLAP", "150")
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64")
)

RETRIEVAL_TOP_K = int(
    os.getenv("RAG_RETRIEVAL_TOP_K", "4")
)

RETRIEVAL_FETCH_K = int(
    os.getenv("RAG_RETRIEVAL_FETCH_K", "12")
)

RETRIEVAL_LAMBDA_MULT = float(
    os.getenv("RAG_RETRIEVAL_LAMBDA_MULT", "0.7")
)


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"缺少环境变量：{name}；"
            f"请检查 {ENV_FILE}"
        )

    return value
