"""This module contains some unit tests for the helper functions in the utility module
"""

import numpy as np
import pytest

from utilities import flat, gaussian_likelihood

@pytest.fixture
def sinus_with_gaussian_noise():
    
    x = np.linspace(0, 5, 200)

    return (
        zip(x, np.sin(x) + np.random.randn(*x.shape)*0),
        lambda x, a, b: a * np.sin(b * x)
    )


@pytest.mark.parametrize(("l", "u"), [(0, 1), (-1, 1)])
def test_flat_zero_outside(l, u):

    assert flat((l + u)/ 2, lower=l, upper=u) !=0
    assert flat(l-1, lower=l, upper=u) == 0
    assert flat(u+1, lower=l, upper=u) == 0
    assert np.all(flat(np.linspace(l, u, 10), lower=l, upper=u) == 1/(u-l))




def test_gaussian_likelihood(sinus_with_gaussian_noise):
    data, func = sinus_with_gaussian_noise

    a, b = np.meshgrid([0, 1, 3], [1, 5, 6, 7])

    p = gaussian_likelihood(data=data, f=func, a=a, b=b)

    assert p.shape == a.shape

    # check normalisation
    assert np.all(p < 1)
    assert np.all(p > 0) 

    # check maximum is for 1*sin(1*x)
    index = np.argmax(p)
    assert (a.flatten()[index] == 1) 
    assert (b.flatten()[index] == 1)

