
# coding: utf-8
import numpy as np
from distance import great_circle
from core import cross_section as conley_cross_section
from core import (CutoffError, get_kernel_fn,
                  _neighbors_to_sparse_uniform, _neighbors_to_sparse_nonuniform)
from numpy.testing import assert_allclose
from hypothesis import given, assume
from hypothesis.strategies import floats, integers, one_of, composite, just, sampled_from
from hypothesis.extra.numpy import arrays
from geopy.distance import EARTH_RADIUS

EPSILON = np.sqrt(np.finfo(float).eps)  # 1.4901161193847656e-08
POSSIBLE_CUTOFFS = floats(min_value = EPSILON, max_value = EARTH_RADIUS * np.pi)
RUN_SLOW_TESTS = False
KNOWN_KERNELS = {'bartlett', 'epanechnikov', 'uniform',
                 # 'biweight', 'quartic', 'tricube', 'triweight',
                 'cosine'}


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
    betahat = bread @ X.T @ y  # '@' is matrix multiplication, equivalent to np.dot
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

        #         k x 1       1 x n        1 x 1
        XeeXh = (((X_i @ row_of_ones * residuals_i) *
                  (column_of_ones @ (residuals.T * window.T))) @ X)
        #                 k x 1                1 x n            n x k
        meat_matrix += XeeXh
    meat_matrix = meat_matrix / N

    sandwich = N * (bread.T @ meat_matrix @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    return se


def test_quakes():
    quakes_y, quakes_X, quakes, quakes_lat_long, _, _, _, cutoff = get_quakes_data()

    # correct_results = conley_unfancy(quakes_y, quakes_X, quakes_lat_long, cutoff)
    correct_results = np.array((108.723235, 19.187791)).reshape(-1, 1)  # faster testing
    fast_results = conley_cross_section("depth ~ mag", quakes,
                                        quakes_lat_long, cutoff)
    assert_allclose(correct_results, fast_results)


@given(POSSIBLE_CUTOFFS)
def test_quakes_random_cutoff(cutoff):
    if RUN_SLOW_TESTS:
        quakes_y, quakes_X, quakes, quakes_lat_long, _, _, _, cutoff = get_quakes_data()

        correct_results = conley_unfancy(quakes_y, quakes_X, quakes_lat_long, cutoff)
        try:
            fast_results = conley_cross_section("depth ~ mag", quakes,
                                                quakes_lat_long, cutoff)
        except CutoffError:
            assume(False)  # This cutoff was too big for the dataset.
        assert_allclose(correct_results, fast_results)

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
    N = 3  # really small example
    K = 1
    # TODO: uncomment these lines instead for more thorough testing:
    # (N is number of rows, K is number of covariates)
    # N = draw(integers(min_value = 1, max_value = 200))
    # K = draw(integers(min_value = 1, max_value = min(max(N, 2) - 1, 30)))
    latitudes = draw(arrays(np.float, (N, 1), elements = floats(
        min_value = -90, max_value = 90)))
    longitudes = draw(arrays(np.float, (N, 1), elements = floats(
        min_value = -180 + EPSILON, max_value = 180)))
    lat_long = np.hstack((latitudes, longitudes))
    # TODO: raise a warning when there's only one location
    assume(unique_rows(lat_long).shape[0] > 2)
    numbers = one_of(floats(min_value = -1 / EPSILON ** 2, max_value = 1 / EPSILON ** 2),
                     integers())
    y = draw(arrays(np.float, (N, 1), elements = numbers))
    X = draw(arrays(np.float, (N, K), elements = numbers))
    assume(np.logical_or(np.abs(X) > EPSILON, X == 0).all())
    assume(np.logical_or(np.abs(y) > EPSILON, y == 0).all())
    cutoff = draw(POSSIBLE_CUTOFFS)
    return (y, X, lat_long, cutoff)


@composite
def generate_geographic_data_with_nan(draw):
    N = 3  # really small example
    K = 1
    # TODO: uncomment these lines instead for more thorough testing:
    # (N is number of rows, K is number of covariates)
    # N = draw(integers(min_value = 1, max_value = 200))
    # K = draw(integers(min_value = 1, max_value = min(max(N, 2) - 1, 30)))

    latitudes = draw(arrays(np.float, (N, 1), elements = floats(
        min_value = -90, max_value = 90)))
    longitudes = draw(arrays(np.float, (N, 1), elements = floats(
        min_value = -180 + EPSILON, max_value = 180)))
    lat_long = np.hstack((latitudes, longitudes))
    # TODO: raise a warning when there's only one location
    assume(unique_rows(lat_long).shape[0] > 2)
    numbers = one_of(floats(min_value = -1 / EPSILON ** 2, max_value = 1 / EPSILON ** 2),
                     integers(), just(np.nan), just(np.inf), just(-np.inf))
    y = draw(arrays(np.float, (N, 1), elements = numbers))
    X = draw(arrays(np.float, (N, K), elements = numbers))
    assume(np.logical_or(np.abs(X) > EPSILON, X == 0).all())
    assume(np.logical_or(np.abs(y) > EPSILON, y == 0).all())
    cutoff = draw(POSSIBLE_CUTOFFS)
    return (y, X, lat_long, cutoff)


def using_random_data(packed_data):
    """Base function to run tests with randomized data.

    Will be decorated by Hypothesis, using arrays with and without nans / infinity.
    """
    # TODO: make things numerically accurate!
    # This test doesn't pass yet; there are issues with numerical accuracy.
    # Since I'm not great with numerical programming, I don't know whether my reference
    # code or my real code is more correct.

    y, X, lat_long, cutoff = packed_data  # necessary to pass the tuple in like this
    try:
        correct_results = conley_unfancy(y, X, lat_long, cutoff)
    except np.linalg.linalg.LinAlgError:
        # Don't worry about these failures:
        # - LinAlgError happens when Hypothesis comes up with a singular matrix.
        assume(False)
    assume(not np.isnan(correct_results).any())
    # This assert really should be true, just checking I haven't done anything dumb above
    assert all(correct_results >= 0)

    # TODO: I think the following block can be replaced by something like
    # conley_cross_section((y, X), data=None, ...)
    # This is really ugly, but I want to be able to use patsy for conley_cross_section,
    # and I only have numpy arrays here, so I need to create a named, combined array,
    # then pass both the names and the combined array.
    x_names = ['x_' + str(xcol) for xcol in range(X.shape[1])]
    data_names = ['y', *x_names]
    dt = {'names': data_names,
          'formats': np.repeat(np.float, len(data_names))}
    data = np.empty(y.shape[0], dtype = dt)
    data['y'] = y[:, 0]
    for xcol, x_name in enumerate(x_names):
        data[x_name] = X[:, xcol]
    data_formula = 'y ~ 0 + ' + ' + '.join(x_names)
    # Formula looks like y ~ 0 + x_0 + x_1 + x_2 ...
    # The zero prevents patsy from adding an intercept.
    try:
        fast_results = conley_cross_section(data_formula, data, lat_long,
                                            cutoff, kernel = 'uniform')
    except (CutoffError, np.linalg.linalg.LinAlgError):
        # CutoffError happens when Hypothesis chooses a cutoff large enough to make every
        # point a neighbor of every other.
        # LinAlgError happens when the X matrix doesn't have full rank (slightly
        # different test than the LinAlgError above, I think).
        assume(False)

    # Don't worry if correct_results has a very small value and fast_results gives nan
    fast_results_nans = np.isnan(fast_results)
    correct_results_small = correct_results < EPSILON
    correct_small_and_fast_nan = np.logical_and(fast_results_nans, correct_results_small)
    assume(not correct_small_and_fast_nan.any())
    # assert our real claim:
    assert_allclose(correct_results, fast_results, rtol = EPSILON, atol = EPSILON)


# TODO: Run this test
@given(generate_geographic_data_no_nan())
def NOtest_random_data_no_nan(packed_data):
    using_random_data(packed_data)


# TODO: Run this test
@given(generate_geographic_data_with_nan())
def NOtest_random_data_with_nan(packed_data):
    using_random_data(packed_data)


# from scipy.sparse import diags
def get_quakes_data():
    from ball_tree import BallTree
    from patsy import dmatrices
    import feather
    quakes = feather.read_dataframe('tests/datasets/quakes.feather')
    quakes_lat = quakes['lat'].values.reshape(-1, 1)
    # Subtract 180 because they've done 0 to 360.  See:
    # https://stackoverflow.com/questions/19879746/why-are-datasetquakes-longtitude-values-above-180
    quakes_long = quakes['long'].values.reshape(-1, 1) - 180
    quakes_lat_long = np.hstack((quakes_lat, quakes_long))
    cutoff = 100

    balltree = BallTree(quakes_lat_long, metric = 'greatcircle')
    neighbors, distances = balltree.query_radius(
        quakes_lat_long, r = cutoff, return_distance = True)

    y, X = dmatrices(data=quakes, formula_like='depth ~ mag')
    betahat, _, rank, _ = np.linalg.lstsq(X, y)
    if rank != X.shape[1]:
        raise np.linalg.LinAlgError('X matrix is not full rank!')
    del rank
    residuals = (y - X @ betahat)
    return y, X, quakes, quakes_lat_long, residuals, neighbors, distances, cutoff



@given(sampled_from(KNOWN_KERNELS))
def test_neighbors_to_sparse(kernel):
    if RUN_SLOW_TESTS:
        assume(kernel != 'uniform')
        _, _, _, _, _, neighbors, distances, cutoff = get_quakes_data()

        from scipy.sparse import find
        neighbors_sparse_withdistance = _neighbors_to_sparse_nonuniform(
            neighbors, kernel, distances, cutoff)
        kernel_fn = get_kernel_fn(kernel)
        for i in range(neighbors.shape[0]):
            # get the indexes that will sort the row
            neighbors_row_argsort = np.argsort(neighbors[i])
            neighbors_row_sorted = neighbors[i][neighbors_row_argsort]

            distance_row_sorted = kernel_fn(distances[i][neighbors_row_argsort], cutoff)

            # test that the neighbor indexes are the same
            np.testing.assert_equal(find(neighbors_sparse_withdistance.getrow(i))[1],
                                    neighbors_row_sorted)
            # test that the distance weight values are the same
            np.testing.assert_equal(find(neighbors_sparse_withdistance.getrow(i))[2],
                                    distance_row_sorted)


def test_neighbors_to_sparse_uniform():
    _, _, _, _, _, neighbors, distances, cutoff = get_quakes_data()

    from scipy.sparse import find
    neighbors_sparse_nodistance = _neighbors_to_sparse_uniform(neighbors)
    for i in range(neighbors.shape[0]):
        # get the indexes that will sort the row
        neighbors_row_argsort = np.argsort(neighbors[i])
        neighbors_row_sorted = neighbors[i][neighbors_row_argsort]
        # test that the neighbor indexes are the same
        np.testing.assert_equal(find(neighbors_sparse_nodistance.getrow(i))[1],
                                neighbors_row_sorted)


@given(sampled_from(KNOWN_KERNELS))
def test_sparse_mult(kernel):
    from scipy.sparse import diags
    _, X, _, _, residuals, neighbors, distances, cutoff = get_quakes_data()

    N = X.shape[0]
    k = X.shape[1]

    meat_matrix = np.zeros((k, k))
    row_of_ones = np.ones((1, N))
    column_of_ones = np.ones((k, 1))
    if kernel == 'uniform':
        neighbors_sp = _neighbors_to_sparse_uniform(neighbors)
    else:
        neighbors_sp = _neighbors_to_sparse_nonuniform(neighbors, kernel,
                                                       distances, cutoff)

    neighbors_dense = neighbors_sp.toarray()
    for i in range(N):
        window = neighbors_dense[i, :]
        X_i = X[i, ].reshape(-1, 1)
        residuals_i = residuals[i, ].reshape(-1, 1)

        #         k x 1       1 x n        1 x 1
        XeeXh = (((X_i @ row_of_ones * residuals_i) *
                  (column_of_ones @ (residuals.T * window.T))) @ X)
        #                 k x 1                1 x n            n x k
        meat_matrix += XeeXh
    correct = meat_matrix / N

    # I want element-wise multiplication of the residuals vector by the neighbors
    # weights matrix. Sparse matrices don't have element-wise multiplication, but
    # it's equivalent to cast the residuals as a sparse diagonal matrix, then do
    # matrix multiplication. The diags function isn't smart enough to convert an
    # (N, 1) array to a (N,) array, so manually reshape. Then, unlike numpy arrays,
    # with sparse matrices, '*' means matrix multiplication, NOT element-wise.
    resid_diag_matrix = diags(residuals.reshape(-1), offsets = 0, shape = (N, N))
    resid_x_neighbors = resid_diag_matrix * neighbors_sp * resid_diag_matrix

    proposed = (X.T @ resid_x_neighbors @ X) / N
    np.testing.assert_allclose(correct, proposed)

    bread = np.linalg.inv(X.T @ X)
    correct_sandwich = N * (bread.T @ correct @ bread)
    correct_se = np.sqrt(np.diag(correct_sandwich)).reshape(-1, 1)
    proposed_sandwich = N * (bread.T @ proposed @ bread)
    proposed_se = np.sqrt(np.diag(proposed_sandwich)).reshape(-1, 1)
    np.testing.assert_allclose(correct_se, proposed_se)
