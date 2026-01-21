from doc_parser.spiders.base_spider import BaseDocsSpider, SequentialSpiderMixin, ParallelSpiderMixin
from doc_parser.spiders_configs import (
    # NumPy
    NumpyReferenceConfig,
    NumpyUserConfig,
    # Pandas
    PandasReferenceConfig,
    PandasUserConfig,
    # Scipy
    ScipyReferenceConfig,
    ScipyUserConfig,
    # Matplotlib
    MatplotlibExamplesConfig,
    MatplotlibReferenceConfig,
    MatplotlibTutorialConfig,
    MatplotlibUserConfig,
    # PyTorch
    TorchDevNotesConfig,
    TorchReferenceConfig,
    # Scikit-learn
    SklearnReferenceConfig,
    SklearnUserConfig,
    # TensorFlow
    TensorflowReferenceConfig,
    TensorflowUserConfig,
)


# SEQUENTIAL SPIDERS

class NumpyDocsSpiderUser(SequentialSpiderMixin, BaseDocsSpider):
    name = NumpyUserConfig.name
    allowed_domains = NumpyUserConfig.allowed_domains
    start_urls = NumpyUserConfig.start_urls
    stop_url = NumpyUserConfig.stop_url


class NumpyDocsSpiderReference(SequentialSpiderMixin, BaseDocsSpider):
    name = NumpyReferenceConfig.name
    allowed_domains = NumpyReferenceConfig.allowed_domains
    start_urls = NumpyReferenceConfig.start_urls
    stop_url = NumpyReferenceConfig.stop_url


class PandasDocsSpiderUser(SequentialSpiderMixin, BaseDocsSpider):
    name = PandasUserConfig.name
    allowed_domains = PandasUserConfig.allowed_domains
    start_urls = PandasUserConfig.start_urls
    stop_url = PandasUserConfig.stop_url


class PandasDocsSpiderReference(SequentialSpiderMixin, BaseDocsSpider):
    name = PandasReferenceConfig.name
    allowed_domains = PandasReferenceConfig.allowed_domains
    start_urls = PandasReferenceConfig.start_urls
    stop_url = PandasReferenceConfig.stop_url


class ScipyDocsSpiderUser(SequentialSpiderMixin, BaseDocsSpider):
    name = ScipyUserConfig.name
    allowed_domains = ScipyUserConfig.allowed_domains
    start_urls = ScipyUserConfig.start_urls
    stop_url = ScipyUserConfig.stop_url


class ScipyDocsSpiderReference(SequentialSpiderMixin, BaseDocsSpider):
    name = ScipyReferenceConfig.name
    allowed_domains = ScipyReferenceConfig.allowed_domains
    start_urls = ScipyReferenceConfig.start_urls
    stop_url = ScipyReferenceConfig.stop_url


# PARALLEL SPIDERS

class MatplotlibDocsSpiderUser(ParallelSpiderMixin, BaseDocsSpider):
    name = MatplotlibUserConfig.name
    allowed_domains = MatplotlibUserConfig.allowed_domains
    start_urls = MatplotlibUserConfig.start_urls
    page_count = 0


class MatplotlibDocsSpiderReference(ParallelSpiderMixin, BaseDocsSpider):
    name = MatplotlibReferenceConfig.name
    allowed_domains = MatplotlibReferenceConfig.allowed_domains
    start_urls = MatplotlibReferenceConfig.start_urls
    page_count = 0


class MatplotlibDocsSpiderTutorial(ParallelSpiderMixin, BaseDocsSpider):
    name = MatplotlibTutorialConfig.name
    allowed_domains = MatplotlibTutorialConfig.allowed_domains
    start_urls = MatplotlibTutorialConfig.start_urls
    page_count = 0


class MatplotlibDocsSpiderExamples(ParallelSpiderMixin, BaseDocsSpider):
    name = MatplotlibExamplesConfig.name
    allowed_domains = MatplotlibExamplesConfig.allowed_domains
    start_urls = MatplotlibExamplesConfig.start_urls
    page_count = 0


class TorchDocsSpiderReference(ParallelSpiderMixin, BaseDocsSpider):
    name = TorchReferenceConfig.name
    allowed_domains = TorchReferenceConfig.allowed_domains
    start_urls = TorchReferenceConfig.start_urls
    page_count = 0


class TorchDocsSpiderDevNotes(ParallelSpiderMixin, BaseDocsSpider):
    name = TorchDevNotesConfig.name
    allowed_domains = TorchDevNotesConfig.allowed_domains
    start_urls = TorchDevNotesConfig.start_urls
    page_count = 0


class SklearnDocsSpiderUser(ParallelSpiderMixin, BaseDocsSpider):
    name = SklearnUserConfig.name
    allowed_domains = SklearnUserConfig.allowed_domains
    start_urls = SklearnUserConfig.start_urls
    page_count = 0


class SklearnDocsSpiderReference(ParallelSpiderMixin, BaseDocsSpider):
    name = SklearnReferenceConfig.name
    allowed_domains = SklearnReferenceConfig.allowed_domains
    start_urls = SklearnReferenceConfig.start_urls
    page_count = 0


class TensorflowDocsSpiderUser(ParallelSpiderMixin, BaseDocsSpider):
    name = TensorflowUserConfig.name
    allowed_domains = TensorflowUserConfig.allowed_domains
    start_urls = TensorflowUserConfig.start_urls
    
    article_selector = 'div.devsite-article-body'
    links_selector = 'ul.devsite-nav-list[menu="_book"] a[href]::attr(href)'
    
    page_count = -1


class TensorflowDocsSpiderReference(ParallelSpiderMixin, BaseDocsSpider):
    name = TensorflowReferenceConfig.name
    allowed_domains = TensorflowReferenceConfig.allowed_domains
    start_urls = TensorflowReferenceConfig.start_urls
    
    article_selector = 'div.devsite-article-body'
    links_selector = 'ul.devsite-nav-list[menu="_book"] a[href]::attr(href)'
    
    page_count = -1
