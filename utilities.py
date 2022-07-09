"""helpers such as various distributions etc. """

import numpy as np

def flat(x, lower, upper):
    try:
        len(x)
        r = np.zeros_like(x)
        L = (x >= lower) & (x <= upper)
        r[L] = 1/(upper - lower)
    except TypeError:
        if x >= lower and x <= upper:
            r = 1/(upper - lower)
        else:
            r = 0
    return r
    

def truncated_jeffreys(x, lower, upper, c=1):

    r = flat(x, lower=lower, upper=upper)
    try:
        r[r!=0] = c/x[r!=0]
    except:
        r = c/x if r != 0 else 0

    return r



def gaussian_log_likelihood(data, f, sigma=1, **model_kwargs):
    """given a gaussian error with variance sigma for the data data
    this gives the log likelihood for the data given the model f 
    with the parameters model_kwargs.
    p(data|func(model_kwargs), sigma, I).

    Data must be an iterable of the shape zip(x_vector, y_vector).
    """
    exponent = -0.5 * np.sum([
        (y-f(x, **model_kwargs))**2 for x, y in data
    ], axis=0) / sigma**2 
    
    normalization = 1/(sigma * np.sqrt(2 * np.pi))
    return exponent + np.log(normalization)

def gaussian_likelihood(data, f, sigma=1, **model_kwargs):
    return np.exp(gaussian_log_likelihood(data=data, f=f, sigma=sigma, **model_kwargs))
