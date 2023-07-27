from qutip import *
import numpy as np
from scipy import optimize
from bayesian import *

from scipy import stats
from scipy.linalg import expm
from scipy.stats import norm
from time import time
import ultranest
import logging

from numpy import exp, cos, cosh, sqrt


def ultranest_sampling(traj,delta_max = 5.):
    parameters = ['delta']
    gamma = 1.
    omega = 1.
    njumps = 48

    def classical_estimator(data: np.array) -> np.array:
        """Returns the classical estimator for the parameter. Only works in 1D, to infer delta"""
        tf = np.sum(data)
        return 0.5 * np.sqrt(np.abs(( 4 * tf * gamma * omega**2 - 8 * njumps * omega**2- njumps * gamma**2 )/ njumps) )

    delta_classical = classical_estimator(traj)

    ## truncated gaussian prior
    myclip_low = 0.0
    myclip_high = np.inf
    loc = delta_classical
    scale = 0.5
    low, high = (myclip_low - loc) / scale, (myclip_high - loc) / scale
    truncated_gaussian_prior = stats.truncnorm(a=low, b=high, loc=loc, scale=scale)

    # follow the inverse cumulative distribution to get the parameter sample given a cube interval [0,1] corresponding to the quantile
    # https://johannesbuchner.github.io/UltraNest/priors.html#id1
    def prior_transform_truncated_gaussian(quantile_cube):
        # the argument, cube, consists of values from 0 to 1 representing the quantiles
        # we have to convert them to physical scales by using the inverse cumulative distribution 

        params = quantile_cube.copy()
        # the parameter correspond to the specific value of the quantile through the inverse cumulative distribution
        params[0] = truncated_gaussian_prior.ppf(quantile_cube[0])
        return params

    # then we need a log likelihood
    def log_likelihood(params):
        delta, = params
        # compute log likelihood for all jumps
        #loglike = np.log(likelihood_fun(delta, traj)+1e-300)  # avoid getting stuck at -inf
        loglike = log_likelihood_fun_Analytical(delta, traj) 
        return loglike

    # create the sampler
    
    sampler = ultranest.ReactiveNestedSampler(parameters, log_likelihood, prior_transform_truncated_gaussian)

    logger = logging.getLogger("ultranest")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)

    result = sampler.run(min_num_live_points=100,viz_callback=False,show_status=False)

    samples = samples_nest=result["samples"]
    median_sample = result['posterior']['median'][0]
    mean_sample = result['posterior']['mean'][0]
    map_sample = result['maximum_likelihood']['point'][0]
    stdev_sample = result["posterior"]["stdev"][0]


    return samples, median_sample, mean_sample, map_sample, stdev_sample