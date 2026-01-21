from scrapy import Spider
from scrapy.selector import SelectorList
from urllib.parse import urlparse
from pathlib import PurePosixPath

from doc_parser.items import DocParseItem
from doc_parser.typing import ItemType, ResponseType

from doc_parser.spiders_configs import (
    NumpyReferenceConfig,
    NumpyUserConfig,
    PandasReferenceConfig,
    PandasUserConfig,
    ScipyReferenceConfig,
    ScipyUserConfig,
)

from doc_parser.constants import EXCLUDED_SELECTORS, BLOCK_ELEMENTS

class NumpyDocsSpiderUser(Spider):

    name = NumpyUserConfig.name
    allowed_domains = NumpyUserConfig.allowed_domains
    start_urls = NumpyUserConfig.start_urls
    stop_url = NumpyUserConfig.stop_url

    page_count = -1

    def parse(self, response: ResponseType) -> ItemType:
        yield from self.parse_section(response)

        if response.url == self.stop_url:
            self.logger.info(
                f"Достигнут целевой URL: {self.stop_url} - завершение работы")
            return

        next_page = response.css('a.right-next::attr(href)').get()
        if next_page:
            self.logger.info(f"Переход на следующую страницу: {next_page}")
            yield response.follow(next_page, callback=self.parse)
        else:
            self.logger.info("Следующая страница не найдена - завершение работы")

    def parse_section(self, response: ResponseType) -> ItemType:
        self.page_count += 1

        path = urlparse(response.url).path
        page_title = PurePosixPath(path).stem
        self.logger.info(f"Парсинг страницы {self.page_count}: {page_title}")

        article = response.css('article.bd-article')

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

            # li с p — контейнер
            if tag == 'li' and el.css('p'):
                continue

            if tag == 'pre':
                text = ''.join(el.css('::text').getall())

            elif tag == 'table':
                rows = []

                for tr in el.xpath('.//tr'):
                    cells = tr.xpath('./th | ./td')
                    row = []

                    for cell in cells:
                        cell_text = ' '.join(cell.css('::text').getall()).strip()
                        row.append(cell_text)

                    if row:
                        rows.append('\t'.join(row))

                text = '\n'.join(rows)
            
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

    def parse_dl(self, dl):
        blocks = []

        for child in dl.xpath('./dt | ./dd'):
            tag = child.root.tag

            if tag == 'dt':
                text = ' '.join(child.css('::text').getall()).strip()
                if text:
                    blocks.append(text)

            elif tag == 'dd':
                # Сначала обрабатываем вложенные dl
                nested_dl = child.xpath('./dl')
                if nested_dl:
                    for ndl in nested_dl:
                        blocks.extend(self.parse_dl(ndl).splitlines())
                
                # Теперь обрабатываем весь остальной контент внутри dd
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
    
    def parse_example(self, article: SelectorList):
        examples = []
        for code_blok in article.xpath('//pre'):
            code = ''.join(code_blok.css('::text').getall())
            if code.strip():
                examples.append(code)
        return examples







class NumpyDocsSpiderReference(NumpyDocsSpiderUser):

    name = NumpyReferenceConfig.name
    allowed_domains = NumpyReferenceConfig.allowed_domains
    start_urls = NumpyReferenceConfig.start_urls
    stop_url = NumpyReferenceConfig.stop_url

class PandasDocsSpiderUser(NumpyDocsSpiderUser):

    name = PandasUserConfig.name
    allowed_domains = PandasUserConfig.allowed_domains
    start_urls = PandasUserConfig.start_urls
    stop_url = PandasUserConfig.stop_url


class PandasDocsSpiderReference(NumpyDocsSpiderUser):

    name = PandasReferenceConfig.name
    allowed_domains = PandasReferenceConfig.allowed_domains
    start_urls = PandasReferenceConfig.start_urls
    stop_url = PandasReferenceConfig.stop_url


class ScipyDocsSpiderUser(NumpyDocsSpiderUser):

    name = ScipyUserConfig.name
    allowed_domains = ScipyUserConfig.allowed_domains
    start_urls = ScipyUserConfig.start_urls
    stop_url = ScipyUserConfig.stop_url


class ScipyDocsSpiderReference(NumpyDocsSpiderUser):

    name = ScipyReferenceConfig.name
    allowed_domains = ScipyReferenceConfig.allowed_domains
    start_urls = ScipyReferenceConfig.start_urls
    stop_url = ScipyReferenceConfig.stop_url
