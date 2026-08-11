from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE


def create_text_splitter(
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            " ",
            "",
        ],
    )


def split_documents(
    documents: list[Document],
) -> list[Document]:
    splitter = create_text_splitter()
    chunks = splitter.split_documents(documents)

    valid_chunks = []

    for chunk_index, chunk in enumerate(chunks):
        content = chunk.page_content.strip()

        if not content:
            continue

        metadata = chunk.metadata.copy()
        metadata["chunk_index"] = chunk_index

        valid_chunk = Document(
            page_content=content,
            metadata=metadata,
        )

        valid_chunks.append(valid_chunk)

    return valid_chunks
