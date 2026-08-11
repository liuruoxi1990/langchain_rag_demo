from langchain_openai import OpenAIEmbeddings


class QueryVectorizer:
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
    ) -> None:
        self.embeddings = embeddings

    def vectorize(self, question: str) -> list[float]:
        question = question.strip()

        if not question:
            raise ValueError("用户问题不能为空")

        vector = self.embeddings.embed_query(question)

        if not vector:
            raise RuntimeError(
                "Embedding 接口返回空问题向量"
            )

        return vector
