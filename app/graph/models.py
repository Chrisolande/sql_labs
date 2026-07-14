from langchain.chat_models import init_chat_model

from app.core.config.settings import settings


def get_fast_model():
    return init_chat_model(model="gemini-2.5-flash", api_key=settings.gemini_api_key)


def get_frontier_model():
    return init_chat_model(
        model="gemini-3.1-flash-lite", api_key=settings.gemini_api_key
    )
