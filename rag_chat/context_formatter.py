from langchain_core.documents import Document


def format_context(
    documents: list[Document],
) -> str:
    parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "未知来源",
        )

        page = document.metadata.get("page")
        content = document.page_content.strip()

        if isinstance(page, int):
            location = f"第 {page + 1} 页"
        else:
            location = "页码未知"

        part = (
            f"[{index}]\n"
            f"来源：{source}\n"
            f"位置：{location}\n"
            f"内容：\n{content}"
        )

        parts.append(part)

    return "\n\n".join(parts)


def format_sources(
    documents: list[Document],
) -> list[str]:
    sources = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "未知来源",
        )

        page = document.metadata.get("page")

        if isinstance(page, int):
            description = (
                f"[{index}] {source}，"
                f"第 {page + 1} 页"
            )
        else:
            description = f"[{index}] {source}"

        sources.append(description)

    return sources
