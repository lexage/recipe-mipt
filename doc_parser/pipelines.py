import re
import string
from scrapy import Item, Spider

from doc_parser.constants import (BASE_DIR, ENCODING, RESULTS_DIR)


def normalize_text(text):
    if not text:
        return ''
    text = text.lower().strip()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '-', text)
    text = text.replace(' ', '_')
    return text

def remove_duplicate_content(raw_text: str) -> str:
    lines = raw_text.splitlines()

    if len(lines) < 10:
        return raw_text

    for window_size in [10, 7, 5, 3]:
        if len(lines) < window_size * 2:
            continue

        reference_lines = lines[:window_size]
        for start_pos in range(window_size, len(lines) - window_size):
            candidate_lines = lines[start_pos : start_pos + window_size]
            if reference_lines != candidate_lines:
                continue

            verification_size = min(window_size * 2, len(lines) - start_pos, start_pos)
            if verification_size <= window_size:
                continue

            extended_ref = lines[:verification_size]
            extended_candidate = lines[start_pos : start_pos + verification_size]
            if extended_ref == extended_candidate:
                return "\n".join(lines[:start_pos])

    return raw_text


class DocsParsePipeline:
    res_dir = BASE_DIR / RESULTS_DIR

    def open_spider(self, spider: Spider) -> None:
        self.res_dir = self.res_dir / spider.name
        self.res_dir.mkdir(parents=True, exist_ok=True)

    def process_item(self, item: Item, spider: Spider) -> Item:
        if not item['title']:
            return
        
        article_name = f"{item['number']}_" + normalize_text(item['title'])
        
        article_save_path = self.res_dir / article_name
        article_save_path.mkdir(parents=True, exist_ok=True)

        article_content = remove_duplicate_content(item['content'])

        for i, example_content in enumerate(item["examples"]):
            
            example_name = "example_" + str(i)

            article_content = article_content.replace(
                example_content.removesuffix('\n'),  
                '<' + example_name + '>', 
                1
                )
            
            with open(article_save_path / (example_name + ".txt") , 'w', encoding=ENCODING) as f:
                f.write(example_content)

        with open(article_save_path / (article_name + ".txt") , 'w', encoding=ENCODING) as f:
            f.write(article_content)
