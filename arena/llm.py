from abc import abstractmethod
from typing import Optional
import openai
from pydantic import BaseModel
from .models import Message
from .config import settings


class LLMResponse(BaseModel):
    """Response from an LLM."""
    content: str
    usage: Optional[dict[str, int]] = None


class BaseLLM(BaseModel):
    """Base class for LLM implementations."""

    model_name: str = settings.default_model
    temperature: float = settings.temperature
    max_tokens: int = settings.max_tokens

    client: Optional[openai.OpenAI] = openai.OpenAI(api_key=settings.openai_api_key)

    @property
    def is_valid(self) -> bool:
        """Check if LLM is properly configured."""
        return bool(self.client and self.model_name and self.temperature and self.max_tokens)

    def validate(self) -> None:
        """Validate LLM configuration."""
        if not self.client:
            raise ValueError("LLM client not initialized")
        if not self.model_name:
            raise ValueError("Model name not specified") 
        if not isinstance(self.temperature, (int, float)) or not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be a float between 0 and 1")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer")

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True
    
    @abstractmethod
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""

    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Generate a response using OpenAI API."""
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return LLMResponse(content=content, usage=usage)
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return LLMResponse(
                content="I apologize, but I'm having trouble connecting to the AI service. Please try again.",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(model_name: Optional[str] = None, **kwargs) -> BaseLLM:
        """Create an LLM instance."""
        model_name = model_name or settings.default_model

        try:
            llm = OpenAILLM(model_name=model_name, **kwargs)
            llm.validate()
            return llm
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI LLM: {e}")
