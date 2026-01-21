from scrapy import Spider
from scrapy.selector import SelectorList
from urllib.parse import urlparse
from pathlib import PurePosixPath

from doc_parser.items import DocParseItem
from doc_parser.typing import ItemType, ResponseType

from doc_parser.spiders_configs import (
    MatplotlibExamplesConfig,
    MatplotlibReferenceConfig,
    MatplotlibTutorialConfig,
    MatplotlibUserConfig,
    TorchDevNotesConfig,
    TorchReferenceConfig,
    SklearnReferenceConfig,
    SklearnUserConfig
)

from doc_parser.constants import EXCLUDED_SELECTORS, BLOCK_ELEMENTS

class MatplotlibDocsSpiderUser(Spider):
    
    name = MatplotlibUserConfig.name
    allowed_domains = MatplotlibUserConfig.allowed_domains
    start_urls = MatplotlibUserConfig.start_urls
    
    page_count = 0

    def parse(self, response: ResponseType) -> ItemType:
        links = response.css('.bd-toc-item .toctree-l1 a.reference.internal, .bd-toc-item .toctree-l2 a.reference.internal')
        
        self.logger.info(f"Найдено {len(links)} ссылок в боковом меню")
        
        yield from response.follow_all(links, callback=self.parse_section)

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
            if code.strip() and (code not in examples):
                examples.append(code)
        return examples

class MatplotlibDocsSpiderReference(MatplotlibDocsSpiderUser):

    name = MatplotlibReferenceConfig.name
    allowed_domains = MatplotlibReferenceConfig.allowed_domains
    start_urls = MatplotlibReferenceConfig.start_urls


class MatplotlibDocsSpiderTutorial(MatplotlibDocsSpiderUser):

    name = MatplotlibTutorialConfig.name
    allowed_domains = MatplotlibTutorialConfig.allowed_domains
    start_urls = MatplotlibTutorialConfig.start_urls


class MatplotlibDocsSpiderExamples(MatplotlibDocsSpiderUser):

    name = MatplotlibExamplesConfig.name
    allowed_domains = MatplotlibExamplesConfig.allowed_domains
    start_urls = MatplotlibExamplesConfig.start_urls


class TorchDocsSpiderReference(MatplotlibDocsSpiderUser):

    name = TorchReferenceConfig.name
    allowed_domains = TorchReferenceConfig.allowed_domains
    start_urls = TorchReferenceConfig.start_urls


class TorchDocsSpiderDevNotes(MatplotlibDocsSpiderUser):

    name = TorchDevNotesConfig.name
    allowed_domains = TorchDevNotesConfig.allowed_domains
    start_urls = TorchDevNotesConfig.start_urls


class SklearnDocsSpiderUser(MatplotlibDocsSpiderUser):

    name = SklearnUserConfig.name
    allowed_domains = SklearnUserConfig.allowed_domains
    start_urls = SklearnUserConfig.start_urls


class SklearnDocsSpiderReference(MatplotlibDocsSpiderUser):

    name = SklearnReferenceConfig.name
    allowed_domains = SklearnReferenceConfig.allowed_domains
    start_urls = SklearnReferenceConfig.start_urls