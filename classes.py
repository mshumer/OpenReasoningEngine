from typing import TypedDict, Optional

# Model Config Type (model name, temperature, top_p, max_tokens)
class ModelConfig(TypedDict):
    model: str
    api_key: Optional[str]
    api_url: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]