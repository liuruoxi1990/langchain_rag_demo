from pathlib import Path

from rag_chat.answer_generator import AnswerGenerator
from rag_chat.chroma_retriever import ChromaRetriever
from rag_chat.context_formatter import (
    format_context,
    format_sources,
)
from rag_chat.query_vectorizer import QueryVectorizer
from utils.config import (
    ARK_BASE_URL,
    ARK_EMBEDDING_MODEL,
    CHROMA_DIR,
    require_env,
)
from vector_init.ark_embedding import create_embeddings
from vector_init.vector_chroma_writer import (
    create_vector_store,
)


class RAGChatApplication:
    def __init__(self) -> None:
        self._check_database_path()

        # 第一步使用的 Embedding 客户端。
        self.embeddings = create_embeddings()

        # 用户问题向量化。
        self.query_vectorizer = QueryVectorizer(
            self.embeddings
        )

        # 加载 Chroma 向量数据库。
        self.vector_store = create_vector_store(
            self.embeddings
        )

        self._check_database_records()

        # Chroma 相似度检索。
        self.retriever = ChromaRetriever(
            self.vector_store
        )

        # Prompt 构造、LLM 调用和答案生成。
        self.answer_generator = AnswerGenerator()

    def _check_database_path(self) -> None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"向量库不存在：{CHROMA_DIR.resolve()}\n"
                "请先执行 python rag_vector_init.py"
            )

    def _check_database_records(self) -> None:
        record_count = (
            self.vector_store
            ._collection
            .count()
        )

        if record_count == 0:
            raise RuntimeError(
                "Chroma 向量库中没有数据。\n"
                "请先执行 python rag_vector_init.py"
            )

        print(
            f"Chroma 加载成功，记录数："
            f"{record_count}"
        )

    def ask(self, question: str) -> tuple[str, list[str]]:
        question = question.strip()

        if not question:
            raise ValueError("问题不能为空")

        # 第一步：问题向量化。
        print("\n[1/5] 正在将问题转换为向量……")

        query_vector = self.query_vectorizer.vectorize(
            question
        )

        print(
            f"问题向量化完成，维度："
            f"{len(query_vector)}"
        )

        # 第二步：Chroma 相似度检索。
        print("[2/5] 正在检索 Chroma……")

        documents = self.retriever.retrieve(
            query_vector
        )

        print(
            f"检索完成，得到 "
            f"{len(documents)} 个文本切片"
        )

        if not documents:
            return "根据当前知识库无法确定。", []

        # 第三步：整理相关文本切片。
        print("[3/5] 正在整理参考资料……")

        context = format_context(documents)
        sources = format_sources(documents)

        # 第四步：构造 Prompt。
        # 第五步：调用方舟模型生成回答。
        print("[4/5] 正在构造 Prompt……")
        print("[5/5] 正在调用火山方舟模型……")

        answer = self.answer_generator.generate(
            question=question,
            context=context,
        )

        return answer, sources


def print_startup_information() -> None:
    chat_model = require_env("ARK_CHAT_MODEL")

    print("\nRAG 问答系统启动成功")
    print(f"Base URL：{ARK_BASE_URL}")
    print(f"Embedding 模型：{ARK_EMBEDDING_MODEL}")
    print(f"对话模型：{chat_model}")
    print(f"向量库：{CHROMA_DIR.resolve()}")
    print("输入 exit、quit 或 q 退出。")


def main() -> None:
    print("正在启动 RAG 问答系统……")

    application = RAGChatApplication()

    print_startup_information()

    while True:
        try:
            question = input("\n问题：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n程序已退出。")
            break

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("程序已退出。")
            break

        if not question:
            continue

        try:
            answer, sources = application.ask(question)

            print("\n回答：")
            print(answer)

            if sources:
                print("\n检索来源：")

                for source in sources:
                    print(source)

        except Exception as exc:
            print("\n问答失败")
            print(f"错误类型：{type(exc).__name__}")
            print(f"错误信息：{exc}")


if __name__ == "__main__":
    main()
