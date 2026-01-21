from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = 'results'
ENCODING = 'utf-8'

EXCLUDED_SELECTORS = [
            'script', 'style', 'nav', 'header', 'footer', 
            '.hidden', '[style*="display:none"]', '[style*="display: none"]', 
            '.devsite-table-wrapper' , 'div.admonition.seealso',
]

BLOCK_ELEMENTS = 'p, h1, h2, h3, h4, h5, h6, li, pre'
