import sys
from pathlib import Path
from transformers import GPT2Tokenizer

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Document
from src.filtering.exactsubstr.main import ExactSubstrFiltrator


def test_1():
    
    texts = [
        'Кошка сидит на коврике. Свежий воздух полезен для здоровья. Маша ушла гулять.',
        'Маша ушла гулять.',
        'Оля играет в настольную игру с друзьями. Настольная игра оказалась очень интересной, всем нравится. Кошка сидит на коврике.',
        'Оля играет в настольную игру с друзьями.',
        'Настольные игры классные.'
    ]

    documents = [
        Document(id=f'doc_{i+1}', text=text, source='local')
        for i, text in enumerate(texts)
    ]
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    filtrator = ExactSubstrFiltrator(
                    threshold=20,
                    enable_bytes=True,
                    enable_tokenizer=False,
                    tokenizer=None)
    
    filtrate_documents = filtrator.apply_documents(documents)
    
    print(*filtrate_documents, sep='\n\n')
    


def test_2():
    
    text_1 = """Hermite interpolation based INVersion of CDF (HINV) #

    Required: CDF

    Optional: PDF, dPDF

    Speed:

    Set-up: (very) slow

    Sampling: (very) fast

    HINV is a variant of numerical inversion, where the inverse CDF is approximated using
    Hermite interpolation, i.e., the interval [0,1] is split into several intervals and
    in each interval the inverse CDF is approximated by polynomials constructed by means
    of values of the CDF and PDF at interval boundaries. This makes it possible to improve
    the accuracy by splitting a particular interval without recomputations in unaffected
    intervals. Three types of splines are implemented: linear, cubic, and quintic
    interpolation. For linear interpolation only the CDF is required. Cubic interpolation
    also requires PDF and quintic interpolation PDF and its derivative.

    These splines have to be computed in a setup step. However, it only works for
    distributions with bounded domain; for distributions with unbounded domain the tails
    are chopped off such that the probability for the tail regions is small compared to
    the given u-resolution.

    The method is not exact, as it only produces random variates of the approximated
    distribution. Nevertheless, the maximal numerical error in “u-direction” (i.e.
    |U   -   CDF(X)|  where  X  is the approximate percentile corresponding to the
    quantile  U  i.e.  X   =   approx_ppf(U) ) can be set to the
    required resolution (within machine precision). Notice that very small values of
    the u-resolution are possible but may increase the cost for the setup step.

    NumericalInverseHermite  approximates the inverse of a continuous
    statistical distribution’s CDF with a Hermite spline. Order of the
    hermite spline can be specified by passing the  order  parameter.

    As described in  [ 1 ] , it begins by evaluating the distribution’s PDF and
    CDF at a mesh of quantiles  x  within the distribution’s support.
    It uses the results to fit a Hermite spline  H  such that
    H(p)   ==   x , where  p  is the array of percentiles corresponding
    with the quantiles  x . Therefore, the spline approximates the inverse
    of the distribution’s CDF to machine precision at the percentiles  p ,
    but typically, the spline will not be as accurate at the midpoints between
    the percentile points:

    p_mid = (p[:-1] + p[1:])/2

    so the mesh of quantiles is refined as needed to reduce the maximum
    “u-error”:

    u_error = np.max(np.abs(dist.cdf(H(p_mid)) - p_mid))

    below the specified tolerance  u_resolution . Refinement stops when the required
    tolerance is achieved or when the number of mesh intervals after the next
    refinement could exceed the maximum allowed number of intervals (100000).

    >>> import numpy as np
    >>> from scipy.stats.sampling import NumericalInverseHermite
    >>> from scipy.stats import norm, genexpon
    >>> from scipy.special import ndtr

    To create a generator to sample from the standard normal distribution, do:

    >>> class StandardNormal:
    ...     def pdf(self, x):
    ...        return 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
    ...     def cdf(self, x):
    ...        return ndtr(x)
    ...
    >>> dist = StandardNormal()
    >>> urng = np.random.default_rng()
    >>> rng = NumericalInverseHermite(dist, random_state=urng)

    The  NumericalInverseHermite  has a method that approximates the PPF of the
    distribution.

    >>> rng = NumericalInverseHermite(dist)
    >>> p = np.linspace(0.01, 0.99, 99) # percentiles from 1% to 99%
    >>> np.allclose(rng.ppf(p), norm.ppf(p))
    True

    Depending on the implementation of the distribution’s random sampling
    method, the random variates generated may be nearly identical, given
    the same random state.

    >>> dist = genexpon(9, 16, 3)
    >>> rng = NumericalInverseHermite(dist)
    >>> # `seed` ensures identical random streams are used by each `rvs` method
    >>> seed = 500072020
    >>> rvs1 = dist.rvs(size=100, random_state=np.random.default_rng(seed))
    >>> rvs2 = rng.rvs(size=100, random_state=np.random.default_rng(seed))
    >>> np.allclose(rvs1, rvs2)
    True

    To check that the random variates closely follow the given distribution, we can
    look at its histogram:

    >>> import matplotlib.pyplot as plt
    >>> dist = StandardNormal()
    >>> urng = np.random.default_rng()
    >>> rng = NumericalInverseHermite(dist, random_state=urng)
    >>> rvs = rng.rvs(10000)
    >>> x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, 1000)
    >>> fx = norm.pdf(x)
    >>> plt.plot(x, fx, 'r-', lw=2, label='true distribution')
    >>> plt.hist(rvs, bins=20, density=True, alpha=0.8, label='random variates')
    >>> plt.xlabel('x')
    >>> plt.ylabel('PDF(x)')
    >>> plt.title('Numerical Inverse Hermite Samples')
    >>> plt.legend()
    >>> plt.show()
    """
    
    text_2 = """Trapezoidal Distribution #

    Two shape parameters  \(c\in[0,1], d\in[0, 1]\)  giving the distances to the
    first and second modes as a percentage of the total extent of
    the non-zero portion. The location parameter is the start of the non-
    zero portion, and the scale-parameter is the width of the non-zero
    portion. In standard form we have  \(x\in\left[0,1\right].\)

    Implementation:  scipy.stats.trapezoid
    
    To check that the random variates closely follow the given distribution, we can
    look at its histogram:

    >>> import matplotlib.pyplot as plt
    >>> dist = StandardNormal()
    >>> urng = np.random.default_rng()
    >>> rng = NumericalInverseHermite(dist, random_state=urng)
    >>> rvs = rng.rvs(10000)
    >>> x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, 1000)
    >>> fx = norm.pdf(x)
    >>> plt.plot(x, fx, 'r-', lw=2, label='true distribution')
    >>> plt.hist(rvs, bins=20, density=True, alpha=0.8, label='random variates')
    >>> plt.xlabel('x')
    >>> plt.ylabel('PDF(x)')
    >>> plt.title('Numerical Inverse Hermite Samples')
    >>> plt.legend()
    >>> plt.show()
    """
    
    texts = [text_1, text_2]

    documents = [
        Document(id=f'doc_{i+1}', text=text, source='local')
        for i, text in enumerate(texts)
    ]
    
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    filtrator = ExactSubstrFiltrator(
                    threshold=50,
                    enable_bytes=True,
                    enable_tokenizer=False,
                    tokenizer=None)
    
    filtrate_documents = filtrator.apply_documents(documents)
    
    print(*filtrate_documents, sep='\n\n')


if __name__=="__main__":
    
    test_1()
    test_2()