

# coding: utf-8
import numpy as np
from distance import great_circle #, vincenty
import feather
from core import cross_section as conley_cross_section
from core import CutoffError
from numpy.testing import assert_array_almost_equal
from hypothesis import given, assume, note
from hypothesis.strategies import floats, tuples, integers, one_of, composite
from hypothesis.extra.numpy import arrays
from geopy.distance import EARTH_RADIUS

EPSILON = np.sqrt(np.finfo(float).eps) # 1.4901161193847656e-08
POSSIBLE_CUTOFFS = floats(min_value = EPSILON, max_value = EARTH_RADIUS * np.pi)
RUN_SLOW_TESTS = False

def great_circle_one_to_many(latlong_array, latlong_point):
    """Find the great-circle distance from each point in an array to another,
    specific point."""
    assert latlong_point.shape == (2,)
    assert latlong_array.shape[1] == 2
    N = latlong_array.shape[0]
    dists = np.empty((N, 1))
    for i, latlong_one_pt in enumerate(latlong_array):
        dists[i] = great_circle(latlong_one_pt, latlong_point)
    return dists


def conley_unfancy(y, X, lat_long, cutoff):
    N = y.shape[0]
    k = X.shape[1]
    bread = np.linalg.inv(X.T @ X)  # 'bread' in the sandwich-estimator sense

    # Run OLS to get residuals
    betahat =  bread @ X.T @ y  # '@' is matrix multiplication, equivalent to np.dot
    residuals = y - X @ betahat
    meat_matrix = np.zeros((k, k))
    row_of_ones = np.ones((1, N))
    column_of_ones = np.ones((k, 1))
    # every_point_is_a_neighbor_of_every_other = True
    for i in range(N):
        dist = great_circle_one_to_many(lat_long, lat_long[i])

        window = dist <= cutoff
        # if not all(window):
        #     every_point_is_a_neighbor_of_every_other = False
        X_i = X[i, ].reshape(-1, 1)
        residuals_i = residuals[i, ].reshape(-1, 1)
        XeeXh = ((X_i @ row_of_ones * residuals_i) * (column_of_ones @ (residuals.T * window.T))) @ X
                    # k x 1       1 x n        1 x 1             k x 1                1 x n        n x k
        meat_matrix += XeeXh
    # if every_point_is_a_neighbor_of_every_other:
    #     raise CutoffError("Every point is a neighbor of every other. You must use a smaller cutoff value.")
    meat_matrix = meat_matrix / N

    sandwich = N * (bread.T @ meat_matrix @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    # results = np.hstack((betahat, se))
    return se

def test_quakes():
    quakes = feather.read_dataframe('tests/datasets/quakes.feather')
    quakes_lat = quakes['lat'].reshape(-1, 1)
    quakes_long = quakes['long'].reshape(-1, 1)
    quakes_lat_long = np.hstack((quakes_lat, quakes_long))
    quakes_y = quakes['depth'].reshape(-1, 1)  # make a (N,) vector into a (N,1) array
    mag_col = quakes['mag'].reshape(-1, 1)
    quakes_X = np.hstack((np.ones_like(mag_col), mag_col))
    quakes_cutoff = 100

    #correct_results = conley_unfancy(quakes_y, quakes_X, quakes_lat_long, quakes_cutoff)
    correct_results = np.array((108.723235, 19.187791)).reshape(-1, 1)  # faster testing
    fast_results = conley_cross_section(quakes_y, quakes_X,
        quakes_lat_long, quakes_cutoff)
    assert_array_almost_equal(correct_results, fast_results)

@given(POSSIBLE_CUTOFFS)
def test_quakes_random_cutoff(quakes_cutoff):
    if RUN_SLOW_TESTS:
        quakes = feather.read_dataframe('tests/datasets/quakes.feather')
        quakes_lat = quakes['lat'].reshape(-1, 1)
        quakes_long = quakes['long'].reshape(-1, 1)
        quakes_lat_long = np.hstack((quakes_lat, quakes_long))
        quakes_y = quakes['depth'].reshape(-1, 1)  # make a (N,) vector into a (N,1) array
        mag_col = quakes['mag'].reshape(-1, 1)
        quakes_X = np.hstack((np.ones_like(mag_col), mag_col))
        correct_results = conley_unfancy(quakes_y, quakes_X, quakes_lat_long, quakes_cutoff)
        try:
            fast_results = conley_cross_section(quakes_y, quakes_X,
                quakes_lat_long, quakes_cutoff)
        except CutoffError:
            assume(False)  # This cutoff was too big for the dataset.
        assert_array_almost_equal(correct_results, fast_results)

def test_new_testspatial():
    new_testspatial = feather.read_dataframe('tests/datasets/new_testspatial.feather')

#
#
# def rnd_len_arrays(dtype, min_len=0, max_len=3, elements=None):
#     lengths = integers(min_value=min_len, max_value=max_len)
#     return lengths.flatmap(lambda n: arrays(dtype, n, elements=elements))


def unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    unique_a = a[idx]
    return(unique_a)

@composite
def generate_geographic_data_no_nan(draw):
    # N = draw(integers(min_value = 1, max_value = 200))  # number of points (rows)
    # K = draw(integers(min_value = 1, max_value = min(max(N, 2) - 1, 30)))  # number of covariates
    N = 3  # really small example
    K = 1
    latitudes  = draw(arrays(np.float, (N, 1), elements =
        floats(min_value = -90, max_value = 90)))
    longitudes = draw(arrays(np.float, (N, 1), elements =
        floats(min_value = -180 + EPSILON, max_value = 180)))
    lat_long = np.hstack((latitudes, longitudes))
    assume(unique_rows(lat_long).shape[0] > 2)  # TODO: raise a warning when there's only one location
    # speed things up by not checking for nans
    # xy_min = -9999999999999
    # xy_max = abs(xy_min)
    numbers = one_of(floats(min_value = -1/np.finfo(float).eps, max_value = 1/np.finfo(float).eps), integers())
    # y = draw(arrays(np.float, (N, 1), elements = one_of(floats(allow_nan = False, allow_infinity = False), integers())))
    # X = draw(arrays(np.float, (N, K), elements = one_of(floats(allow_nan = False, allow_infinity = False), integers())))
    y = draw(arrays(np.float, (N, 1), elements = numbers))
    X = draw(arrays(np.float, (N, 1), elements = numbers))
    assume(np.logical_or(np.abs(X) > EPSILON, X == 0).all())
    assume(np.logical_or(np.abs(y) > EPSILON, y == 0).all())
    cutoff = draw(POSSIBLE_CUTOFFS)
    return (y, X, lat_long, cutoff)

@given(generate_geographic_data_no_nan())
def notest_random_data(packed_data):
    y, X, lat_long, cutoff = packed_data  # seems to be necessary to pass the tuple in like this
    try:
        correct_results = conley_unfancy(y, X, lat_long, cutoff)
    except np.linalg.linalg.LinAlgError:
    # Don't worry about these failures:
    # - LinAlgError happens when Hypothesis comes up with a singular matrix.
        assume(False)
    assume(not np.isnan(correct_results).any())
    assert all(correct_results >= 0)  # really should be true, just checking I haven't done anything dumb
    try:
        fast_results = conley_cross_section(y, X, lat_long, cutoff)
    except CutoffError:
        # CutoffError happens when Hypothesis chooses a cutoff large enough to make every point a neighbor of every other.
        assume(False)

    # Don't worry if correct_results has a very small value and fast_results gives nan
    fast_results_nans = np.isnan(fast_results)
    correct_results_small = correct_results < EPSILON
    correct_small_and_fast_nan = np.logical_and(fast_results_nans, correct_results_small)
    assume(not correct_small_and_fast_nan.any())

    # assert our real claim:
    assert_array_almost_equal(correct_results, fast_results)

#TODO test with nans
