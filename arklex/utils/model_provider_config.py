from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings


def get_huggingface_llm(model, **kwargs):
    llm = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        **kwargs
    )
    return ChatHuggingFace(llm=llm)

LLM_PROVIDERS = ["openai", "gemini", "anthropic", "huggingface", "ollama"]

PROVIDER_MAP = {
    "anthropic": ChatAnthropic,
    "gemini": ChatGoogleGenerativeAI,
    "openai": ChatOpenAI,
    "huggingface": get_huggingface_llm,
    "ollama": ChatOllama
}

PROVIDER_EMBEDDINGS = {
    "anthropic": HuggingFaceEmbeddings,
    "gemini": GoogleGenerativeAIEmbeddings,
    "openai": OpenAIEmbeddings ,
    "huggingface": HuggingFaceEmbeddings,
    "ollama": OllamaEmbeddings
}
PROVIDER_EMBEDDING_MODELS = {
    "anthropic": "sentence-transformers/sentence-t5-base",
    "gemini": "models/embedding-001",
    "openai": "text-embedding-ada-002",
    "huggingface": "sentence-transformers/all-mpnet-base-v2",
    "ollama": "nomic-embed-text",
}