from scrapy import Spider
from scrapy.selector import SelectorList
from urllib.parse import urlparse
from pathlib import PurePosixPath
from scrapy import Item
from scrapy.http import Response

from doc_parser.items import DocParseItem
from doc_parser.constants import EXCLUDED_SELECTORS


class BaseDocsSpider(Spider):
    
    article_selector = 'article.bd-article'
    
    page_count = -1

    def get_page_title(self, response: Response) -> str:
        path = urlparse(response.url).path
        return PurePosixPath(path).stem

    def parse_section(self, response: Response) -> Item:
        self.page_count += 1
        
        page_title = self.get_page_title(response)
        
        self.logger.info(f"Парсинг страницы {self.page_count}: {page_title}")
        
        article = response.css(self.article_selector)

        # удаляем явно ненужные элементы
        for selector in EXCLUDED_SELECTORS:
            for element in article.css(selector):
                element.drop()

        paragraphs = []
        
        elements = article.xpath(
            './/*['
            'self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 '
            'or (self::pre and not(ancestor::dd)) '
            'or self::table '
            'or self::dl[not(ancestor::dl)] '
            'or ((self::p or self::li) '
            'and not(ancestor::table) '
            'and not(ancestor::dd) '
            'and not(ancestor::dt))'
            ']'
        )

        for el in elements:
            tag = el.root.tag

            if tag == 'li' and el.css('p'):
                continue

            if tag == 'pre':
                text = ''.join(el.css('::text').getall())

            elif tag == 'table':
                text = self.parse_table(el)
            
            elif tag == 'dl':
                text = self.parse_dl(el)

            else:
                text = ' '.join(el.css('::text').getall())

            text = text.strip()
            if text:
                paragraphs.append(text)

        full_text = '\n\n'.join(paragraphs)

        examples = self.parse_example(article)

        yield DocParseItem({
            'title': page_title,
            'content': full_text,
            'url': response.url,
            'number': self.page_count,
            'examples': examples,
        })

    def parse_table(self, table_element) -> str:
        rows = []

        for tr in table_element.xpath('.//tr'):
            cells = tr.xpath('./th | ./td')
            row = []

            for cell in cells:
                cell_text = ' '.join(cell.css('::text').getall()).strip()
                row.append(cell_text)

            if row:
                rows.append('\t'.join(row))

        return '\n'.join(rows)

    def parse_dl(self, dl) -> str:
        blocks = []

        for child in dl.xpath('./dt | ./dd'):
            tag = child.root.tag

            if tag == 'dt':
                text = ' '.join(child.css('::text').getall()).strip()
                if text:
                    blocks.append(text)

            elif tag == 'dd':
                nested_dl = child.xpath('./dl')
                if nested_dl:
                    for ndl in nested_dl:
                        blocks.extend(self.parse_dl(ndl).splitlines())
                
                other_elements = child.xpath('./*[not(self::dl)]')
                for el in other_elements:
                    el_tag = el.root.tag
                    if el_tag == 'pre':
                        text = ''.join(el.css('::text').getall())
                    else:
                        text = ''.join(el.css('::text').getall())
                    text = text.strip()
                    if text:
                        blocks.append(text)

        return '\n'.join(blocks)
    
        """Извлекает примеры кода из pre блоков с фильтрацией дубликатов."""
        examples = []
        for code_blok in article.xpath('//pre'):
            code = ''.join(code_blok.css('::text').getall())
            if code.strip() and (code not in examples):
                examples.append(code)
        return examples

    def parse_example(self, article: SelectorList) -> list:
        examples = []
        for code_blok in article.xpath('//pre'):
            code = ''.join(code_blok.css('::text').getall())
            if code.strip() and (code not in examples):
                examples.append(code)
        return examples


class SequentialSpiderMixin:
    
    # Должен быть определён в дочернем классе
    stop_url = None
    next_page_selector = 'a.right-next::attr(href)'
    
    def parse(self, response: Response) -> Item:
        """Последовательный проход по страницам."""
        yield from self.parse_section(response)

        if response.url == self.stop_url:
            self.logger.info(
                f"Достигнут целевой URL: {self.stop_url} - завершение работы")
            return

        next_page = response.css(self.next_page_selector).get()
        if next_page:
            self.logger.info(f"Переход на следующую страницу: {next_page}")
            yield response.follow(next_page, callback=self.parse)
        else:
            self.logger.info("Следующая страница не найдена - завершение работы")


class ParallelSpiderMixin:
    
    # Должен быть определён в дочернем классе
    links_selector = '.bd-toc-item .toctree-l1 a.reference.internal, .bd-toc-item .toctree-l2 a.reference.internal'
    
    def parse(self, response: Response) -> Item:
        """Параллельный проход по всем ссылкам."""
        links = response.css(self.links_selector)
        
        self.logger.info(f"Найдено {len(links)} ссылок в боковом меню")
        
        yield from response.follow_all(links, callback=self.parse_section)
