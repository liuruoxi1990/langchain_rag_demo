from rag_chat.ark_chat_model import ArkChatModel
from rag_chat.prompt_builder import PromptBuilder


class AnswerGenerator:
    def __init__(self) -> None:
        self.prompt_builder = PromptBuilder()
        self.chat_model = ArkChatModel()

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )

        response = self.chat_model.invoke(prompt)
        content = response.content

        if isinstance(content, str):
            return content.strip()

        return str(content).strip()
