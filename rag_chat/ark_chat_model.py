from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from utils.config import (
    ARK_BASE_URL,
    require_env,
)


def create_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=require_env("ARK_CHAT_MODEL"),
        api_key=require_env("ARK_API_KEY"),
        base_url=ARK_BASE_URL,
        temperature=0.1,
        timeout=120,
        max_retries=2,
    )


class ArkChatModel:
    def __init__(self) -> None:
        self.model = create_chat_model()

    def invoke(self, prompt) -> AIMessage:
        return self.model.invoke(
            prompt.to_messages()
        )
