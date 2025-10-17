from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import urllib.parse
from typing import List


class WebRequest:
    """"Class for search information in internet """  

    def search_duckduckgo(self, query: str):
        ddgs = DDGS()
        results = ddgs.text(query, max_results=5)
        return [result['body'] for result in results]
    
    def retrieve(self, question):
        search_results = self.search_duckduckgo(question)
        return {"context": "\n\n".join(search_results)}


def duckduckgo_search(query: str, max_results: int = 5) -> List[str]:

    """
    Выполняет поиск через DuckDuckGo и возвращает список описаний.
    Args:
        query: Поисковый запрос
        max_results: Максимальное количество результатов
    
    Returns:
        Список описаний веб-сайтов
    """

    try:
        # Формируем URL для DuckDuckGo
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        # Устанавливаем заголовки чтобы имитировать браузер
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        # Отправляем запрос
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Парсим HTML с помощью BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []

        # Ищем результаты поиска (DuckDuckGo использует такие классы)
        search_results = soup.find_all('div', class_='result')[:max_results]
        
        for result in search_results:
            # Извлекаем описание
            description_elem = result.find('a', class_='result__snippet')
            if description_elem:
                description = description_elem.get_text(strip=True)
                if description:
                    results.append(description)
            
            # Если не нашли описание, пытаемся получить заголовок
            if not results:
                title_elem = result.find('a', class_='result__a')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title:
                        results.append(title)        
        return results[:max_results]

    except Exception as e:
        return [f"Ошибка при выполнении поиска: {str(e)}"]


if __name__ == '__main__':
    test_query = 'Rachmaninoff piano concerto no. 3'
    result = WebRequest().retrieve(test_query)
    result_2 = duckduckgo_search(test_query)
    print(result, result_2, sep='\n\n')
