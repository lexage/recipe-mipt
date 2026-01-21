
BOT_NAME = "doc_parser"

SPIDER_MODULES = ["doc_parser.spiders"]
NEWSPIDER_MODULE = "doc_parser.spiders"

ADDONS = {}

ROBOTSTXT_OBEY = True

FEED_EXPORT_ENCODING = "utf-8"

ITEM_PIPELINES = {
    'doc_parser.pipelines.DocsParsePipeline': 300,
}
