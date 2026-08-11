from utils.config import DATA_DIR

from vector_init.ark_embedding import (
    create_embeddings,
    validate_embedding,
)
from vector_init.document_processor import (
    process_directory,
)
from vector_init.text_chunker import (
    split_documents,
)
from vector_init.text_cleaner import (
    clean_documents,
)
from vector_init.vector_chroma_writer import (
    get_database_path,
    write_documents,
)


def print_chunk_preview(chunks) -> None:
    if not chunks:
        return

    print("\n第一个切片预览：")
    print("-" * 60)
    print(chunks[0].page_content[:500])
    print("-" * 60)


def main() -> None:
    print("开始初始化 RAG 向量知识库")
    print(f"数据目录：{DATA_DIR.resolve()}")

    # 第一步：解析 PDF、TXT 和 Markdown 文档。
    raw_documents = process_directory(DATA_DIR)

    print(
        f"\n解析得到的原始文档单元："
        f"{len(raw_documents)}"
    )

    if not raw_documents:
        raise RuntimeError(
            "没有解析到任何文档"
        )

    # 第二步：清洗并提取有效纯文本。
    cleaned_documents = clean_documents(
        raw_documents
    )

    print(
        f"清洗后的有效文档单元："
        f"{len(cleaned_documents)}"
    )

    if not cleaned_documents:
        raise RuntimeError(
            "文档解析成功，但没有提取到有效文本"
        )

    # 第三步：文本切片。
    chunks = split_documents(
        cleaned_documents
    )

    print(
        f"生成的有效文本切片："
        f"{len(chunks)}"
    )

    if not chunks:
        raise RuntimeError(
            "没有生成有效文本切片"
        )

    print_chunk_preview(chunks)

    # 第四步：创建并验证 Embedding 服务。
    embeddings = create_embeddings()

    dimension = validate_embedding(
        embeddings
    )

    # 第五步：生成向量并写入 Chroma。
    print("\n开始生成向量并写入 Chroma……")

    record_count = write_documents(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("\nRAG 向量知识库初始化完成")
    print(f"原始文档单元：{len(raw_documents)}")
    print(f"有效文档单元：{len(cleaned_documents)}")
    print(f"文本切片：{len(chunks)}")
    print(f"向量维度：{dimension}")
    print(f"Chroma 记录数：{record_count}")
    print(f"向量库位置：{get_database_path()}")


if __name__ == "__main__":
    main()
