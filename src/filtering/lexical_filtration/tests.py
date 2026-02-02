import pytest

from src.agent_constructor.core import Chunk
from filtering.lexical_filtration.simple_lexic_filtrator import (
    FilteringConfig,
    SimpleLexicalFiltrator,
)


@pytest.fixture
def config_for_lexical_filtrator() -> FilteringConfig:
    return FilteringConfig()


@pytest.fixture
def lexical_filtrator() -> SimpleLexicalFiltrator:
    return SimpleLexicalFiltrator()


text1 = """Dunnett’s test#

Dunnett’s test compares the means of multiple experimental groups against a
single control group. In [1], the influence of drugs on blood count
measurements on three groups of animals is investigated.

The following table summarizes the results of the experiment in which two
groups received different drugs, and one group acted as a control. Blood
counts (in millions of cells per cubic millimeter) were recorded:

import numpy as np
control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])
drug_a = np.array([9.76, 8.80, 7.68, 9.36])
drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])

import numpy as np
control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])
drug_a = np.array([9.76, 8.80, 7.68, 9.36])
drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])

import numpy as np
control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])
drug_a = np.array([9.76, 8.80, 7.68, 9.36])
drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])

import numpy as np
control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])
drug_a = np.array([9.76, 8.80, 7.68, 9.36])
drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])

import numpy as np
control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])
drug_a = np.array([9.76, 8.80, 7.68, 9.36])
drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])

We would like to see if the means between any of the groups are
significantly different. First, visually examine a box and whisker plot.

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.boxplot([control, drug_a, drug_b])
ax.set_xticklabels(["Control", "Drug A", "Drug B"])
ax.set_ylabel("mean")
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.boxplot([control, drug_a, drug_b])
ax.set_xticklabels(["Control", "Drug A", "Drug B"])
ax.set_ylabel("mean")
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.boxplot([control, drug_a, drug_b])
ax.set_xticklabels(["Control", "Drug A", "Drug B"])
ax.set_ylabel("mean")
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.boxplot([control, drug_a, drug_b])
ax.set_xticklabels(["Control", "Drug A", "Drug B"])
ax.set_ylabel("mean")
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.boxplot([control, drug_a, drug_b])
ax.set_xticklabels(["Control", "Drug A", "Drug B"])
ax.set_ylabel("mean")
plt.show()

Note the overlapping interquartile ranges of the drug A group and control group
and the apparent separation between the drug B group and control group.

Next, we will use Dunnett's test to assess whether
the difference between group means is significant while controlling the
family-wise error rate: the probability of making any false discoveries.

Let the null hypothesis be that the experimental groups have the same mean as
the control and the alternative be that an experimental group does not have the
same mean as the control. We will consider a 5% family-wise error rate to be
acceptable, and therefore we choose 0.05 as the threshold for significance.

from scipy.stats import dunnett
res = dunnett(drug_a, drug_b, control=control)
res.pvalue




array([0.62009281, 0.00582381])

from scipy.stats import dunnett
res = dunnett(drug_a, drug_b, control=control)
res.pvalue

from scipy.stats import dunnett
res = dunnett(drug_a, drug_b, control=control)
res.pvalue

from scipy.stats import dunnett
res = dunnett(drug_a, drug_b, control=control)
res.pvalue

from scipy.stats import dunnett
res = dunnett(drug_a, drug_b, control=control)
res.pvalue

array([0.62009281, 0.00582381])

array([0.62009281, 0.00582381])

array([0.62009281, 0.00582381])

array([0.62009281, 0.00582381])

The p-value corresponding with the comparison between group A and control
exceeds 0.05, so we do not reject the null hypothesis for that comparison.
However, the p-value corresponding with the comparison between group B and
control is less than 0.05, so we consider the experimental results to be
evidence against the null hypothesis in favor of the alternative: group B has a
different mean than the control group.

References#

Dunnett, Charles W. (1955) “A Multiple Comparison Procedure for Comparing
Several Treatments with a Control.” Journal of the American Statistical
Association, 50:272, 1096-1121. DOI:10.1080/01621459.1955.10501294"""


text2 = """Artists#
Created On: May 04, 2021 | Last Updated On: May 22, 2024
Almost all objects you interact with on a Matplotlib plot are called "Artist"
(and are subclasses of the Artist class).  Figure
and Axes are Artists, and generally contain
Axis Artists and Artists that contain data or annotation information.

Introduction to Artists
Creating Artists
Changing Artist properties
Changing Artist data
Manually adding Artists
Removing Artists

Introduction to Artists
Creating Artists
Changing Artist properties
Changing Artist data
Manually adding Artists
Removing Artists

Creating Artists

Changing Artist properties

Changing Artist data

Manually adding Artists

Removing Artists

Automated color cycle
Optimizing Artists for performance
Paths
Path effects guide
Understanding the extent keyword argument of imshow
Transformations Tutorial

Automated color cycle

Optimizing Artists for performance

Paths

Path effects guide

Understanding the extent keyword argument of imshow

Transformations Tutorial"""


def test_lexical_filtrator(lexical_filtrator: SimpleLexicalFiltrator) -> None:
    chunk_1 = Chunk(text=text1, id="123", doc_id="456")
    chunk_2 = Chunk(text=text2, id="890", doc_id="43830")
    chunks_for_filtering = [chunk_1, chunk_2]
    initial_len_chunk_1 = len(chunk_1.text)
    initial_len_chunk_2 = len(chunk_2.text)
    new_chunks = lexical_filtrator.apply(chunks_for_filtering)
    print(
        "Отфильтрованный текст чанка 1:\n",
        (
            new_chunks[0].text
            if new_chunks[0] is not None
            else "Весь чанк был мусорным"
        ),
    )
    print(
        "Отфильтрованный текст чанка 2:\n",
        (
            new_chunks[1].text
            if new_chunks[1] is not None
            else "Весь чанк был мусорным"
        ),
    )
    assert initial_len_chunk_1 > len(chunk_1.text)
    assert initial_len_chunk_2 > len(chunk_2.text)
    assert "References#" not in chunk_1.text
    assert "Dunnett, Charles W. (1955)" not in chunk_1.text
    assert "DOI:10.1080" not in chunk_1.text
    assert "Last Updated On" not in chunk_2.text
