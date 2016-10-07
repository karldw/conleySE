
# coding: utf-8
import numpy as np
from core import get_kernel_fn
from numpy.testing import assert_allclose
from typedefs import DTYPE
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, one_of, composite

NUMBERS = one_of(floats(allow_infinity = False, allow_nan = False), integers())
EPSILON = np.sqrt(np.finfo(float).eps)  # 1.4901161193847656e-08

# KNOWN_KERNELS = {'bartlett', 'epanechnikov', 'uniform',
#                  'biweight', 'quartic', 'tricube',
#                  'triweight', 'cosine'}
# dists = np.array([-1, 0, 1, 2, 3], dtype = DTYPE)
# cutoff = 3

@composite
def generate_dists_cutoff(draw):
    N = draw(integers(min_value = 0, max_value = 200))
    dists = draw(arrays(DTYPE, (N,), elements = NUMBERS))
    cutoff = draw(floats(min_value = 10 * EPSILON,
                         allow_infinity = False, allow_nan = False))
    dists = dists[np.abs(dists) <= cutoff]
    return (dists, cutoff)


@given(generate_dists_cutoff())
def test_bartlett(dists_and_cutoff):
    dists, cutoff = dists_and_cutoff
    kernel_fn = get_kernel_fn('bartlett')
    claimed_weights = kernel_fn(dists, cutoff)
    correct_weights = 1 - np.abs(dists / cutoff)
    assert_allclose(claimed_weights, correct_weights, atol = 0.00001)


@given(generate_dists_cutoff())
def test_epanechnikov(dists_and_cutoff):
    dists, cutoff = dists_and_cutoff
    kernel_fn = get_kernel_fn('epanechnikov')
    claimed_weights = kernel_fn(dists, cutoff)
    correct_weights = 3 / 4 * (1 - (dists / cutoff)**2) * (np.abs(dists) < cutoff)
    assert_allclose(claimed_weights, correct_weights, atol = 0.00001)


@given(generate_dists_cutoff())
def test_cosine(dists_and_cutoff):
    dists, cutoff = dists_and_cutoff
    kernel_fn = get_kernel_fn('cosine')
    claimed_weights = kernel_fn(dists, cutoff)
    correct_weights = (np.pi / 4 * (np.cos(np.pi / 2 * (dists / cutoff))) *
                       (np.abs(dists) < cutoff))
    assert_allclose(claimed_weights, correct_weights, atol = 0.00001)


# @given(generate_dists_cutoff())
# def test_biweight(dists_and_cutoff):
#     dists, cutoff = dists_and_cutoff
#     kernel_fn = get_kernel_fn('biweight')
#     claimed_weights = kernel_fn(dists, cutoff)
#     correct_weights = (15 / 16 * ((1 - (dists / cutoff)**2) ** 2) *
#                        (np.abs(dists) < cutoff))
#     assert_allclose(claimed_weights, correct_weights, atol = 0.00001)
#
#
# @given(generate_dists_cutoff())
# def test_tricube(dists_and_cutoff):
#     dists, cutoff = dists_and_cutoff
#     kernel_fn = get_kernel_fn('tricube')
#     claimed_weights = kernel_fn(dists, cutoff)
#     correct_weights = (70 / 81 * (1 - np.abs((dists / cutoff))**3)**3 *
#                        (np.abs(dists) < cutoff))
#     assert_allclose(claimed_weights, correct_weights, atol = 0.00001)
#
#
# @given(generate_dists_cutoff())
# def test_triweight(dists_and_cutoff):
#     dists, cutoff = dists_and_cutoff
#     kernel_fn = get_kernel_fn('triweight')
#     claimed_weights = kernel_fn(dists, cutoff)
#     correct_weights = (35 / 32) * (1 - (dists / cutoff)**2)**3 * (np.abs(dists) < cutoff)
#     assert_allclose(claimed_weights, correct_weights, atol = 0.00001)
