from langchain_openai import OpenAIEmbeddings

from utils.config import (
    ARK_BASE_URL,
    ARK_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    require_env,
)


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=ARK_EMBEDDING_MODEL,
        api_key=require_env("ARK_API_KEY"),
        base_url=ARK_BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=EMBEDDING_BATCH_SIZE,
        skip_empty=True,
    )


def validate_embedding(
    embeddings: OpenAIEmbeddings,
) -> int:
    print("\n正在测试 Agent Plan Embedding 接口……")
    print(f"Base URL：{ARK_BASE_URL}")
    print(f"Embedding 模型：{ARK_EMBEDDING_MODEL}")

    try:
        vector = embeddings.embed_query(
            "这是一条向量化接口测试文本。"
        )
    except Exception as exc:
        raise RuntimeError(
            "Agent Plan Embedding 调用失败。\n"
            f"错误类型：{type(exc).__name__}\n"
            f"错误信息：{exc}"
        ) from exc

    if not vector:
        raise RuntimeError(
            "Agent Plan Embedding 接口返回了空向量"
        )

    dimension = len(vector)

    print("Embedding 测试成功")
    print(f"向量维度：{dimension}")

    return dimension
