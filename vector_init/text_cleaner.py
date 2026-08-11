import re

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 合并行内连续空格，但保留换行结构。
    text = re.sub(r"[ \t]+", " ", text)

    # 删除每行首尾空格。
    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    text = "\n".join(lines)

    # 最多保留两个连续换行。
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(
    documents: list[Document],
) -> list[Document]:
    cleaned_documents = []

    for document in documents:
        cleaned_text = clean_text(
            document.page_content
        )

        if not cleaned_text:
            source = document.metadata.get(
                "source",
                "未知来源",
            )

            page = document.metadata.get("page")

            if isinstance(page, int):
                print(
                    f"[跳过] {source} 第 {page + 1} 页"
                    "没有有效文本"
                )
            else:
                print(f"[跳过] 空白文档：{source}")

            continue

        cleaned_document = Document(
            page_content=cleaned_text,
            metadata=document.metadata.copy(),
        )

        cleaned_documents.append(
            cleaned_document
        )

    return cleaned_documents
