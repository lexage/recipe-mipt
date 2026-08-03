import json
import backoff
import openai
import re

from openai import OpenAI


class OpenAIClientWrapper:
    """
    Обертка над OpenAI клиентом с обработкой RateLimitError 
    через экспоненциальный бэкофф.
    """

    def __init__(self, api_key: str = "", base_url: str = "", model_name: str = ""):
        """
        Args:
            api_key (str): API ключ OpenAI.
            base_url (str): альтернативный URL (если используется прокси).
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_info = model_name

    @staticmethod
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def _completions_with_backoff(client: OpenAI, **kwargs):
        """
        Запрос к API OpenAI с экспоненциальным бэкоффом при RateLimitError.
        
        Args:
            client (OpenAI): клиент OpenAI.
            **kwargs: параметры запроса.
        
        Returns:
            OpenAI response object.
        """
        return client.chat.completions.create(**kwargs)

    def get_request(self, content: str, model: str = "gpt-4o-mini") -> str:
        """
        Отправляет запрос в модель OpenAI.
        
        Args:
            content (str): текст запроса (prompt).
            model (str): название модели.
        
        Returns:
            str: ответ модели.
        """
        messages = [{"role": "user", "content": content}]
        request = self._completions_with_backoff(self.client, messages=messages, model=model)
        return request.choices[0].message.content
