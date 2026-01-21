from scrapy import Field, Item

class DocParseItem(Item):
    number = Field()
    title = Field()
    content = Field()
    url = Field()
    examples = Field()
