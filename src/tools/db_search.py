from base_tool import AbstractTool
from typing import Dict, Any


class DBSearchTool(AbstractTool):
    
    def __init__(self, db, top_k):
        
        self.name = "db_search"
        self.description =  f"A tool for extracting information from a database."
        self.db = db
        self.top_k = top_k
    
    def get_schema(self) -> Dict[str, Any]:
        """Возвращает описание инструмента в формате OpenAI Tools API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The text of the query to be searched in the database."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def run(self, query: str) -> str:
        """Извлекает контекст из базы данных"""
        if self.db:
            retrieved = self.db.query(query, self.top_k)
            return retrieved
        