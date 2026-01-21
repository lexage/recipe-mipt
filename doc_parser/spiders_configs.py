from dataclasses import dataclass


@dataclass
class NumpyUserConfig:
    name = 'numpy_user_guide'
    allowed_domains = ['numpy.org']
    start_urls = ['https://numpy.org/doc/stable/user/whatisnumpy.html']
    stop_url = 'https://numpy.org/doc/stable/glossary.html'


@dataclass
class NumpyReferenceConfig:
    name = 'numpy_reference'
    allowed_domains = ['numpy.org']
    start_urls = ['https://numpy.org/doc/stable/reference/index.html']
    stop_url = 'https://numpy.org/doc/stable/reference/swig.testing.html'


@dataclass
class PandasUserConfig:
    name = 'pandas_user_guide'
    allowed_domains = ['pandas.pydata.org']
    start_urls = ['https://pandas.pydata.org/docs/user_guide/index.html']
    stop_url = 'https://pandas.pydata.org/docs/user_guide/cookbook.html'


@dataclass
class PandasReferenceConfig:
    name = 'pandas_reference'
    allowed_domains = ['pandas.pydata.org']
    start_urls = ['https://pandas.pydata.org/docs/reference/index.html']
    stop_url = 'https://pandas.pydata.org/docs/reference/api/pandas.NaT.html'


@dataclass
class ScipyUserConfig:
    name = 'scipy_user_guide'
    allowed_domains = ['docs.scipy.org']
    start_urls = ['https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide']
    stop_url = 'https://docs.scipy.org/doc/scipy/tutorial/thread_safety.html'


@dataclass
class ScipyReferenceConfig:
    name = 'scipy_reference'
    allowed_domains = ['docs.scipy.org']
    start_urls = ['https://docs.scipy.org/doc/scipy/reference/index.html']
    stop_url = 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.EmpiricalDistributionFunction.plot.html'


@dataclass
class TorchReferenceConfig:
    name = 'torch_reference'
    allowed_domains = ['docs.pytorch.org']
    start_urls = ['https://docs.pytorch.org/docs/stable/torch.html']


@dataclass
class TorchDevNotesConfig:
    name = 'torch_dev_notes'
    allowed_domains = ['docs.pytorch.org']
    start_urls = ['https://docs.pytorch.org/docs/stable/notes/amp_examples.html']

@dataclass
class SklearnUserConfig:
    name = 'sklearn_user_guide'
    allowed_domains = ['scikit-learn.org']
    start_urls = ['https://scikit-learn.org/stable/supervised_learning.html']


@dataclass
class SklearnReferenceConfig:
    name = 'sklearn_reference'
    allowed_domains = ['scikit-learn.org']
    start_urls = ['https://scikit-learn.org/stable/api/index.html']


@dataclass
class MatplotlibUserConfig:
    name = 'matplotlib_user_guide'
    allowed_domains = ['matplotlib.org']
    start_urls = ['https://matplotlib.org/stable/users/explain/quick_start.html']


@dataclass
class MatplotlibReferenceConfig:
    name = 'matplotlib_reference'
    allowed_domains = ['matplotlib.org']
    start_urls = ['https://matplotlib.org/stable/api/index.html']


@dataclass
class MatplotlibTutorialConfig:
    name = 'matplotlib_tutorial'
    allowed_domains = ['matplotlib.org']
    start_urls = ['https://matplotlib.org/stable/tutorials/index.html']


@dataclass
class MatplotlibExamplesConfig:
    name = 'matplotlib_examples'
    allowed_domains = ['matplotlib.org']
    start_urls = ['https://matplotlib.org/stable/gallery/index.html']


@dataclass
class TensorflowUserConfig:
    name = 'tensorflow_user_guide'
    allowed_domains = ['www.tensorflow.org']
    start_urls = ['https://www.tensorflow.org/guide']


@dataclass
class TensorflowReferenceConfig:
    name = 'tensorflow_reference'
    allowed_domains = ['www.tensorflow.org']
    start_urls = ['https://www.tensorflow.org/api_docs/python/tf']
