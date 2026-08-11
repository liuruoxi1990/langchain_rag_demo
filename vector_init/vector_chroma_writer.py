import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from utils.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
)


def build_chunk_id(document: Document) -> str:
    source = str(
        document.metadata.get("source", "")
    )

    page = str(
        document.metadata.get("page", "")
    )

    chunk_index = str(
        document.metadata.get("chunk_index", "")
    )

    content = document.page_content.strip()

    raw = (
        f"{source}|{page}|{chunk_index}|{content}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def create_vector_store(
    embeddings: OpenAIEmbeddings,
) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )


def write_documents(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
) -> int:
    if not chunks:
        raise RuntimeError(
            "没有可写入 Chroma 的文本切片"
        )

    vector_store = create_vector_store(
        embeddings
    )

    ids = [
        build_chunk_id(chunk)
        for chunk in chunks
    ]

    batch_size = EMBEDDING_BATCH_SIZE

    for start in range(0, len(chunks), batch_size):
        end = min(
            start + batch_size,
            len(chunks),
        )

        batch_documents = chunks[start:end]
        batch_ids = ids[start:end]

        print(
            f"正在写入切片 {start + 1}～{end}，"
            f"总数 {len(chunks)}"
        )

        try:
            vector_store.add_documents(
                documents=batch_documents,
                ids=batch_ids,
            )
        except Exception as exc:
            raise RuntimeError(
                f"写入第 {start + 1}～{end} 个切片失败。\n"
                f"错误类型：{type(exc).__name__}\n"
                f"错误信息：{exc}"
            ) from exc

    return vector_store._collection.count()


def get_database_path() -> Path:
    return Path(CHROMA_DIR).resolve()
