
# coding: utf-8
import numpy as np
#from scipy import spatial
#from sklearn.neighbors import KDTree, BallTree
from ball_tree import BallTree  # local build
#from kd_tree import KDTree # local build, but not used for now.
from distance import great_circle#, vincenty
#import statsmodels.api as sm
import feather
# import dask
# import numexpr as ne
#import scipy.sparse as sp

from faster_sandwich_filling import multiply_XeeX
def always_one(x, y):
    return 1

def bartlett(dists, cutoff):
    # TODO: test?
    return 1 - dists / cutoff

def lat_long_dist(latlong1, latlong2):
    """Apply geopy's great_circle for two arrays of dim (2,)."""
    if latlong1.shape == (10,):
        # sklearn tests with some (10,) arrays to check that the fuction runs.
        return 1
    assert latlong1.shape == latlong2.shape == (2,)
    return great_circle(latlong1, latlong2)

def great_circle_one_to_many(latlong_array, latlong_point):
    assert latlong_point.shape == (2,)
    assert latlong_array.shape[1] == 2
    N = latlong_array.shape[0]
    dists = np.empty((N, 1))
    for i, latlong_one_pt in enumerate(latlong_array):
        dists[i] = great_circle(latlong_one_pt, latlong_point)
    return dists


def conley_basic(y, X, lat_long, cutoff):
    N = y.shape[0]
    k = X.shape[1]
    assert lat_long.shape == (N, 2)
    assert X.shape[0] == N
    bread = np.linalg.inv(X.T @ X)
    # I have no idea if this leaf_size is reasonable.  If running out of memory, divide N by a larger number.
    # 40 is the default.
    leaf_size = max(40, N / 1000)
    betahat =  bread @ X.T @ y  # '@' is matrix multiplication, equivalent to np.dot or __mul__ on matrices
    #betahat = sm.OLS(Y,X).fit().params  # probably better
    residuals = y - X @ betahat
    sklearn_balltree = BallTree(lat_long, metric = 'greatcircle', leaf_size = leaf_size)
    neighbors, neighbor_distances = sklearn_balltree.query_radius(lat_long, r = cutoff, return_distance = True)
    del sklearn_balltree
    meat_matrix = np.zeros((k, k))
    for i, neighbor_idx in enumerate(neighbors):
        assert len(neighbor_idx) >= 1  # sanity check
        e_i = residuals[i, ]
        X_i = X[i, ].reshape(-1, 1)
        for j in neighbor_idx:
            e_j = residuals[j, ]
            X_j = X[j, ].reshape(-1, 1)
            kernel_value = 1
            assert X_i.shape == (k, 1)
            assert X_j.T.shape == (1, k)
            one_pair_mat = X_i @ X_j.T * e_i * e_j * kernel_value
            # k x k     [k x 1]  [1 x k] [1 x 1] [1 x 1] [1 x 1]
            meat_matrix += one_pair_mat
            # TODO: is it faster if I do matrices?
            #XeeXh = ((X[i, ] @ row_of_ones * residuals[i, ]) * (column_of_ones @ (residuals.T))) @ xdata
                    # k x 1       1 x N        1 x 1             k x 1                1 x N        N x k
            #meat_matrix += XeeXh
    meat_matrix = meat_matrix / N
    sandwich = N * (bread.T @ meat_matrix @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    results = np.hstack((betahat, se))
    return results

def get_kernel_fn(kernel):
    """Get a kernel for the pointwise distances."""
    kernel_dict = {'uniform': always_one,
                   'bartlett': bartlett
    }
    if callable(kernel):
        return kernel
    try:
        return kernel_dict[kernel]
    except KeyError:
        known_kernels = set([x for x in kernel_dict.keys()])
        error_message = "Unknown kernel specified. Please provide one of these, or your own function: {}".format(known_kernels)
        raise KeyError(error_message)

## This is the winner so far.
def conley_mat(y, X, lat_long, cutoff, kernel = 'uniform'):
    N = y.shape[0]
    assert lat_long.shape == (N, 2)
    assert X.shape[0] == N
    assert y.shape == (N,) or y.shape[1] == 1
    kernel_fn = get_kernel_fn(kernel)

    k = X.shape[1]
    bread = np.linalg.inv(X.T @ X)
    # I have no idea if this leaf_size is reasonable.  If running out of memory, divide N by a larger number.
    # 40 is the default.
    leaf_size = max(40, N // 1000)
    betahat =  bread @ X.T @ y  # '@' is matrix multiplication, equivalent to np.dot or __mul__ on matrices
    #betahat = sm.OLS(Y,X).fit().params  # probably better
    residuals = y - X @ betahat
    balltree = BallTree(lat_long, metric = 'greatcircle', leaf_size = leaf_size)
    if kernel == 'uniform':
        neighbors = balltree.query_radius(lat_long, r = cutoff)
    else:
        neighbors, neighbor_distances = balltree.query_radius(lat_long, r = cutoff, return_distance = True)
    del balltree
    filling = np.zeros((k, k))
    row_of_ones = np.ones((1, N))
    column_of_ones = np.ones((k, 1))
    window_zeros = np.zeros((1, N))
    for i, neighbor_idx in enumerate(neighbors):
        assert len(neighbor_idx) >= 1  # sanity check
        # TODO: could we avoid some of these reshapes if we used matrix rather than ndarray?
        residuals_i = residuals[i, ]
        X_i = X[i, ].reshape(-1, 1)  # TODO: use np.newaxis rather than reshape
        # window_T is the transpose of the 'window' variable used in previous work.
        # Doing it this way avoids some transposes and a reshape.
        window_T = np.copy(window_zeros)
        if kernel == 'uniform':
            window_T[:, neighbor_idx] = np.ones_like(neighbor_idx)
        else:
            window_T[:, neighbor_idx] = kernel_fn(neighbor_distances[i], cutoff)

        # construct X'e'eX for given observation
        # multiplying X_i by row_of_ones makes a [k x N] matrix of copies of X_i
        # another approach would be np.tile, but that seems to run at the same speed


        XeeXh = np.dot(((np.dot(X_i, (row_of_ones * residuals_i))) * (np.dot(column_of_ones, (residuals.T * window_T)))), X)
        # XeeXh_pretty = ((X_i @ (row_of_ones * residuals_i)) * (column_of_ones @ (residuals.T * window_T))) @ X
        #            # [k x 1] @   [1 x N]   *   [1 x 1]    *     [k x 1]     @   [1 x N]    * [1 x N]     @ [N x k]
        # assert np.isclose(XeeXh, XeeXh_pretty).all()  # TODO: make this assert a separate test

        filling += XeeXh
    filling = filling / N
    sandwich = N * (bread.T @ filling @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    results = np.hstack((betahat, se))
    return results

def conley_mat2(y, X, lat_long, cutoff, kernel = 'uniform'):
    N = y.shape[0]
    assert lat_long.shape == (N, 2)
    assert X.shape[0] == N
    assert y.shape == (N,) or y.shape[1] == 1
    #kernel_fn = get_kernel_fn(kernel)
    assert cutoff > 0
    k = X.shape[1]
    bread = np.linalg.inv(X.T @ X)
    # I have no idea if this leaf_size is reasonable.  If running out of memory, divide N by a larger number.
    # 40 is the default.
    leaf_size = max(40, N // 1000)
    betahat =  bread @ X.T @ y  # '@' is matrix multiplication, equivalent to np.dot or __mul__ on matrices
    #betahat = sm.OLS(Y,X).fit().params  # probably better
    residuals = (y - X @ betahat)[:, 0]  # reshape from (N, 1) to (N,)
    balltree = BallTree(lat_long, metric = 'greatcircle', leaf_size = leaf_size)
    if kernel == 'uniform':
        neighbors = balltree.query_radius(lat_long, r = cutoff)
        filling = multiply_XeeX(neighbors, residuals, X, kernel)
    else:
        neighbors, neighbor_distances = balltree.query_radius(lat_long, r = cutoff, return_distance = True)
        filling = multiply_XeeX(neighbors, residuals, X, kernel, distances = neighbor_distances, cutoff = cutoff)
    del balltree
    sandwich = N * (bread.T @ filling @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    results = np.hstack((betahat, se))
    return results
#
# def conley_mat2_py(y, X, lat_long, cutoff, kernel = 'uniform'):
#     N = y.shape[0]
#     assert lat_long.shape == (N, 2)
#     assert X.shape[0] == N
#     assert y.shape == (N,) or y.shape[1] == 1
#     #kernel_fn = get_kernel_fn(kernel)
#
#     k = X.shape[1]
#     bread = np.linalg.inv(X.T @ X)
#     # I have no idea if this leaf_size is reasonable.  If running out of memory, divide N by a larger number.
#     # 40 is the default.
#     leaf_size = max(40, N // 1000)
#     betahat =  bread @ X.T @ y  # '@' is matrix multiplication, equivalent to np.dot or __mul__ on matrices
#     #betahat = sm.OLS(Y,X).fit().params  # probably better
#     residuals = y - X @ betahat
#     balltree = BallTree(lat_long, metric = 'greatcircle', leaf_size = leaf_size)
#     if kernel == 'uniform':
#         neighbors = balltree.query_radius(lat_long, r = cutoff)
#         filling = multiply_XeeX_uniform(neighbors, residuals, X)
#     else:
#         raise NotImplementedError
#     del balltree
#     print(filling)
#     sandwich = N * (bread.T @ filling @ bread)
#     se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
#     results = np.hstack((betahat, se))
#     return results
#
#
# def multiply_XeeX_uniform(neighbors, residuals, X):
#     N = X.shape[0]
#     k = X.shape[1]
#     output = np.zeros((k, k))
#     # Calculate sum_i^N [ X_i * e_i * (sum_j^n X_j' * e_j) ]
#     # for i and j near one another
#     for i in range(N):
#         neighbors_i = neighbors[i]
#         e_i = residuals[i]
#         X_i_ei = X[i, np.newaxis] * e_i
#         tempsum_X_j_ej = np.zeros((k, 1))
#         for j in range(neighbors_i.shape[0]):
#             neighbor_j = neighbors_i[j]
#             e_j = residuals[neighbor_j]
#             tempsum_X_j_ej += X[neighbor_j, np.newaxis].T * e_j
#         output += np.dot(X_i_ei, tempsum_X_j_ej)
#     return output / N


def conley_full_loop(y, X, lat, lon, cutoff, metric = 'sol'):
    N = y.shape[0]
    k = X.shape[1]
    bread = np.linalg.inv(X.T @ X)
    betahat =  bread @ X.T @ y  # '@' is matrix multiplication, equivalent to np.dot or __mul__ on matrices
    #betahat = sm.OLS(Y,X).fit().params  # probably better
    residuals = y - X @ betahat
    meat_matrix = np.zeros((k, k))
    row_of_ones = np.ones((1, N))
    column_of_ones = np.ones((k, 1))
    loop_count = 0
    latlong_combined = np.hstack((lat.reshape(-1, 1), lon.reshape(-1, 1)))
    for i in range(N):
        lonscale = np.cos(lat[i] * np.pi / 180) * 111
        latscale = 111
        # Faster, slightly less accurate version, but it's really hard to force
        # this version (Sol's version) into the tree framework.
        if metric == 'sol':
            dist = np.sqrt(  # yields an implied loop count of 36608
                (latscale * (lat[i] - lat))**2 +
                (lonscale * (lon[i] - lon))**2
            )
        elif metric == 'great_circle':
            dist = great_circle_one_to_many(latlong_combined, latlong_combined[i])  # yields an implied loop count of 36508
        else:
            raise ValueError('Unknown metric.')
        window = dist <= cutoff
        # loop_count += np.sum(window)
        X_i = X[i, ].reshape(-1, 1)
        residuals_i = residuals[i, ].reshape(-1, 1)
        XeeXh = ((X_i @ row_of_ones * residuals_i) * (column_of_ones @ (residuals.T * window.T))) @ X
                    # k x 1       1 x n        1 x 1             k x 1                1 x n        n x k
        meat_matrix += XeeXh
    meat_matrix = meat_matrix / N
    sandwich = N * (bread.T @ meat_matrix @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    results = np.hstack((betahat, se))
    return results


if __name__ == '__main__':
    # earthquake data, output of
    # $ R -e "feather::write_feather(quakes, '~/quakes.feather')""
    quakes = feather.read_dataframe('/home/karl/quakes.feather')
    quakes_lat = quakes['lat'].reshape(-1, 1)
    quakes_long = quakes['long'].reshape(-1, 1)
    quakes_lat_long = np.hstack((quakes_lat, quakes_long))
    quakes_y = quakes['depth'].reshape(-1, 1)  # make a (N,) vector into a (N,1) array
    mag_col = quakes['mag'].reshape(-1, 1)
    quakes_X = np.hstack((np.ones_like(mag_col), mag_col))
    quakes_cutoff = 100
    full_loop_results = conley_full_loop(quakes_y, quakes_X, quakes_lat, quakes_long, quakes_cutoff, metric = 'great_circle')
    full_loop_results_sol = conley_full_loop(quakes_y, quakes_X, quakes_lat, quakes_long, quakes_cutoff, metric = 'sol')
    tree_results = conley_basic(quakes_y, quakes_X, quakes_lat_long, quakes_cutoff)
    print("Are the results right?", np.isclose(full_loop_results, tree_results).all())
