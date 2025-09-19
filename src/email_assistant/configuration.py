import os
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(temperature: float = 0.0, **kwargs):
    """Return a configured Gemini LLM.

    Args:
        temperature: The temperature to use for the LLM.
        **kwargs: Additional arguments to pass to the LLM (e.g., model).

    Returns:
        A configured ChatGoogleGenerativeAI instance.
    """
    # Allow explicit model override via kwargs, fallback to env var (2.5 series by default)
    model_name = kwargs.pop("model", os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    # langchain-google-genai expects bare model name (no provider prefix or models/ prefix)
    if ":" in model_name:
        _, model_name = model_name.split(":", 1)
    if model_name.startswith("models/"):
        model_name = model_name.split("/", 1)[1]
    # Gemini now natively supports system messages; disable the legacy conversion layer to
    # avoid the noisy "Convert_system_message_to_human will be deprecated" warning.
    kwargs.setdefault("convert_system_message_to_human", False)
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, **kwargs)
