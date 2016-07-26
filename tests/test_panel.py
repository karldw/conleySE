

# coding: utf-8
import numpy as np
from distance import great_circle #, vincenty
import feather
from core import cross_section as conley_cross_section
from core import panel as conley_panel
from core import CutoffError, parse_lat_long
from numpy.testing import assert_array_almost_equal
from hypothesis import given, assume, note
from hypothesis.strategies import floats, tuples, integers, one_of, composite
from hypothesis.extra.numpy import arrays
from geopy.distance import EARTH_RADIUS
from patsy import dmatrices, dmatrix

EPSILON = np.sqrt(np.finfo(float).eps) # 1.4901161193847656e-08
POSSIBLE_CUTOFFS = floats(min_value = EPSILON, max_value = EARTH_RADIUS * np.pi)
RUN_SLOW_TESTS = True


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


def iterateObs(data_subset, formula, correction_type, cutoff, lat_long):
    """Translated from R -- IterateObs_Fn.R"""
    if correction_type == 'spatial':
        nrow_subset = data_subset.shape[0]
        _, X_subset = dmatrices(formula, data_subset)
        k = X_subset.shape[1]
        lat_long_subset = parse_lat_long(lat_long, data_subset)
        assert lat_long_subset.shape == (data_subset.shape[0], 2)
        XeeXhs = XeeXhC(lat_long = lat_long_subset, cutoff = cutoff, X = X_subset,
                        e = data_subset['residuals'])
    elif correction_type == 'serial':
        raise NotImplementedError()
    else:
        raise ValueError("Invalid correction_type: '{}'".format(correction_type))
    return XeeXhs


    #
    # if (type == "spatial") {
    #         sub_dt <- dt[time == sub_index]
    #         n1 <- nrow(sub_dt)
    #         if(n1 > 1000 & verbose){message(paste("Starting on sub index:", sub_index))}
    #
    #         X <- as.matrix(sub_dt[, eval(Xvars), with = FALSE])
    #         e <- sub_dt[, e]
    #         lat <- sub_dt[, lat]; lon <- sub_dt[, lon]
    #
    #         # If n1 >= 50k obs, then avoiding construction of distance matrix.
    #         # This requires more opeations, but is less memory intensive.
    #         if(n1 < 5 * 10^4) {
    #             XeeXhs <- XeeXhC(cbind(lat, lon), cutoff, X, e, n1, k,
    #                 kernel, dist_fn)
    #         } else {
    #             XeeXhs <- XeeXhC_Lg(cbind(lat, lon), cutoff, X, e, n1, k,
    #                 kernel, dist_fn)
    #         }
    #     ##############################################
    #     } else if(type == "serial") {
    #         sub_dt <- dt[unit == sub_index]
    #         n1 <- nrow(sub_dt)
    #         if(n1 > 1000 & verbose){message(paste("Starting on sub index:", sub_index))}
    #
    #         X <- as.matrix(sub_dt[, eval(Xvars), with = FALSE] )
    #         e <- sub_dt[, e]
    #         times <- sub_dt[, time]
    #
    #         XeeXhs <- TimeDist(times, cutoff, X, e, n1, k)
    #     }
    #
    #     XeeXhs
    # }

def XeeXhC(lat_long, cutoff, X, e):
    """Translated from Rccp file ConleySE.cpp

    But only haversine (great circle) distance and bartlett weighting.

    This function is run on subsets within each time grouping."""
    nrow = lat_long.shape[0]
    k = X.shape[1]
    dmat = np.zeros((nrow, nrow))
    for i in range(nrow):
        dmat[i, i] = 1
        for j in range(i + 1, nrow):
            d = great_circle(lat_long[i], lat_long[j])
            if d > cutoff:
                continue
            weight = 1 - (d / cutoff)
            dmat[i, j] = weight
            dmat[j, i] = weight


    XeeXh = np.zeros((k, k))
    e_mat = np.empty((1, nrow))
    k_mat = np.ones((k, 1))
    e = np.copy(e)  # don't want any of this views nonsense messing with my indexing
    # (i should correspond to rows of e *and* iterations of the loop)
    for i in range(nrow):
        # Note: in arma, '*' is matrix multiplication and '%' is element-wise.
        # In python (>= 3.5), '@' is matrix and '*' is element-wise.

        e_mat.fill(e[i])
        d_row = dmat[i, :] * np.asarray(e.T)
        d_row = d_row.reshape(-1, 1).T
        X_row = X[i, np.newaxis].T
        XeeXh += (X_row @ e_mat * (k_mat @ d_row)) @ X;

    return XeeXh


def conley_panel_unfancy(formula, data, lat_long, group_varname = 'FIPS',
    time_varname = 'year', dist_cutoff = 500, time_cutoff = 5):
    """
    Translated from R -- 'ConleySEs_17June2015.R'

    Testing with bartlett kernel in time and a uniform kernel in space.

    """
    #TODO: IS THIS RIGHT?
    y, X = dmatrices(formula, data)
    time = data[time_varname]
    group = data[group_varname]

    nobs = y.shape[0]
    k_param = X.shape[1]

    betahat, _, rank, _= np.linalg.lstsq(X, y)
    if rank != X.shape[1]:
        raise np.linalg.LinAlgError('X matrix is not full rank!')
    data['residuals'] = (y - X @ betahat)
    # Correct for spatial correlation:
    timeUnique = np.unique(time)
    XeeX = np.zeros((k_param, k_param))
    for t in timeUnique:
        XeeX += iterateObs(data_subset = data[time == t], formula = formula,
            correction_type = "spatial", cutoff = dist_cutoff, lat_long = lat_long)

    bread = np.linalg.inv(X.T @ X) * nobs
    V_spatial = bread @ (XeeX / nobs) @ bread / nobs
    V_spatial = (V_spatial + V_spatial.T) / 2

    # Correct for serial correlation:
    panelUnique = np.unique(group)
    XeeX_serial = np.zeros_like(XeeX)
    for ID in panelUnique:
        XeeX_serial += iterateObs(data_subset = data[group == ID], formula = formula,
            correction_type = "serial", cutoff = time_cutoff, lat_long = lat_long)
    XeeX += XeeX_serial  # seems weird, but this is what they do
    V_spatial_HAC = bread @ (XeeX / nobs) @ bread / nobs
    V_spatial_HAC = (V_spatial_HAC + V_spatial_HAC.T) / 2

    return V_spatial_HAC


def test_new_testspatial():
    """Use the new_testspatial dataset from Thiemo.

    (Exported to a feather dataframe with haven 0.2.0 and feather 0.0.1)
    """
    new_testspatial = feather.read_dataframe('tests/datasets/new_testspatial.feather')

    #TODO: to be comparable with Thiemo, should add year and FIPS fixed effects
    formula = 'EmpClean ~ HDD + unemploymentrate - 1'
    correct_results = conley_panel_unfancy(formula, new_testspatial, lat_long =
        ('latitude', 'longitude'), group_varname = 'FIPS', time_varname = 'year',
        dist_cutoff = 500, time_cutoff = 5)

    fast_results = conley_panel(formula, new_testspatial,  lat_long =
        ('latitude', 'longitude'), time = 'year', group = 'FIPS',  dist_cutoff = 500,
        time_cutoff = 5, dist_kernel = 'uniform', time_kernel = 'bartlett')
