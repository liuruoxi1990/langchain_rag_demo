from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.config import (
    RETRIEVAL_FETCH_K,
    RETRIEVAL_LAMBDA_MULT,
    RETRIEVAL_TOP_K,
)


class ChromaRetriever:
    def __init__(
        self,
        vector_store: Chroma,
    ) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query_vector: list[float],
    ) -> list[Document]:
        if not query_vector:
            raise ValueError("问题向量不能为空")

        documents = (
            self.vector_store
            .max_marginal_relevance_search_by_vector(
                embedding=query_vector,
                k=RETRIEVAL_TOP_K,
                fetch_k=RETRIEVAL_FETCH_K,
                lambda_mult=RETRIEVAL_LAMBDA_MULT,
            )
        )

        return documents
