from typing import Iterable
from scrapy import Item, Spider, crawler, http

CrawlerType = crawler.Crawler
ExceptionType = Exception
ItemType = Item
RequestType = http.Request
ResponseType = http.Response
ResultType = Iterable[ItemType]
SpiderType = Spider
StartRequestType = Iterable[RequestType]
