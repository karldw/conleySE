
# coding: utf-8
import numpy as np
from ball_tree import BallTree  # TODO: force python to only look locally for import

# TODO: rename faster_sandwich_filling to something more informative
from faster_sandwich_filling import CutoffError, GeographyError, get_kernel_fn
from faster_sandwich_filling import multiply_XeeX  # TODO: replace this with sparse ver
import warnings
# import statsmodels.api as sm
from patsy import dmatrices, dmatrix
# from statsmodels.formula.api import ols
from scipy.sparse import coo_matrix
from typedefs import ITYPE, DTYPE


def check_geography(N, lat_long, cutoff):
    # fun fact: 6371.009 * pi, approx 20015 km, is farthest
    # apart two points can be on earth.
    if cutoff <= 0 or cutoff >= 20015:
        raise CutoffError('Please provide a cutoff in km.')
    if lat_long.shape != (N, 2):
        err_msg = ("Your lat_long variable should be an N-by-2 array. The first column"
                   "is latitude, the second is longitude, both measured in degrees. "
                   "Rows represent observations.")
        raise GeographyError(err_msg)
    if not np.isfinite(lat_long).all():
        err_msg = ("Your lat_long variable must be fully populated with real numbers. "
                   "The first column is latitude, the second is longitude, both measured"
                   " in degrees.")
        raise GeographyError(err_msg)
    lat_abs_max = np.max(np.abs(lat_long[:, 0]))
    long_max = np.max(lat_long[:, 1])
    long_min = np.min(lat_long[:, 1])
    if lat_abs_max > 90:
        raise GeographyError("Latitude must be in the range [-90, 90].")
    if long_max > 180 or long_min <= -180:
        raise GeographyError("longitude must be in the range (-180, 180].")
    if lat_abs_max <= np.pi / 2 and long_max <= np.pi and long_min > -np.pi:
        warn_msg = ("Your lat_long variable should be measured in degrees, but it looks"
                    "like you've provided radians – all the latitudes are in [-π/2, π/2]"
                    " and the longitudes are in (-π, π].")
        warnings.warn(warn_msg, UserWarning)


def check_parameters(y, X, lat_long, cutoff):
    N = y.shape[0]
    if X.shape[0] != N:
        raise ValueError("Your y and X arrays must have the same number of rows.")
    if not (y.shape == (N, 1)):
        raise ValueError("Your y array must be of shape (N,1).")
    check_geography(N, lat_long, cutoff)
    return N


def check_parameters_panel(y, X, time, lat_long, cutoff):
    nobs = check_parameters(y, X, lat_long, cutoff)
    if time.shape[0] != nobs:
        err_msg = ("Your time array must have the same number of rows as your X and y "
                   "arrays.")
        raise ValueError(err_msg)
    if time.shape != (nobs,) and time.shape != (nobs, 1):
        raise ValueError("Your time array must have one column.")
    return nobs


def cross_section(formula_like, data, lat_long, cutoff, kernel = 'uniform'):
    """Calculate Conley standard errors for a cross section.

    Parameters
    ----------
    formula_like : string or other Patsy formula
        e.g. "my_y_variable = my_X_var1 + my_X_var2"
        See http://patsy.readthedocs.io/en/latest/formulas.html#formulas for
        details on Patsy formulas.
    data : array-like
        Must contain all the variables referenced in the formula.
    lat_long : array_like, or tuple of names of columns in data
        An N-by-2 array of latitudes (in the first column) and longitudes (in
        the second column). Both latitude and longitude should be measured
        in degrees. Valid longitudes are [-90, 90].  Valid latitudes are
        (-180, 180]. The number of rows should be the same as the rows in data.
    cutoff : number
        The maximum distance over which covariance is possible.
        cutoff must be a positive number in the range (0, 20015).
    kernel : string
        The kernel function to weight the distances by. Valid options are:
        'bartlett', 'triangle', 'epanechnikov', 'quartic', 'biweight' and
        'triweight'. (Bartlett is the same as triangle. Quartic is the same as
        biweight.)
    """
    y, X = dmatrices(formula_like, data, eval_env = 1, NA_action = 'raise')
    # TODO: handle cases where people provide weird formulas?

    lat_long = parse_lat_long(lat_long, data)
    # Raise an exception if the data look funky
    nobs = check_parameters(y, X, lat_long, cutoff)

    # I have no idea if this leaf_size is reasonable.  If running out of memory,
    # divide N by a larger number.
    # 40 is the default.
    leaf_size = max(40, nobs // 1000)
    # TODO: consider a more sophisticated way of calculating residuals (e.g. one that
    # allows for fancy fixed effects)
    betahat, _, rank, _ = np.linalg.lstsq(X, y)
    if rank != X.shape[1]:
        raise np.linalg.LinAlgError('X matrix is not full rank!')
    del rank
    residuals = (y - X @ betahat)
    balltree = BallTree(lat_long, metric = 'greatcircle', leaf_size = leaf_size)
    if kernel == 'uniform':
        neighbors = balltree.query_radius(lat_long, r = cutoff)
        filling = multiply_XeeX(neighbors, residuals, X, kernel)
    else:
        neighbors, neighbor_distances = balltree.query_radius(
            lat_long, r = cutoff, return_distance = True)
        filling = multiply_XeeX(neighbors, residuals, X, kernel,
                                distances = neighbor_distances, cutoff = cutoff)
        del neighbor_distances
    del balltree, neighbors, y, residuals

    bread = np.linalg.inv(X.T @ X)
    sandwich = nobs * (bread.T @ filling @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    return se


def parse_lat_long(lat_long, data):  # noqa: C901
    """Flexibly acquire latitude-longitude array.

    Very important: you should run `check_geography` on the resulting array.

    Options:
        1) `lat_long` is a tuple of names to pull from `data`.
        2) `lat_long` is a tuple of arrays/vectors/DataFrames.
        3) Almost anything else.
    Returns:
        1) An (N, 2) array with columns pulled by name from `data`.
        2) An (N, 2) array with columns stacked horizontally.
        3) Whatever you provided.

    If `lat_long` is a tuple of strings, I assume they're in the order
    (latitude, longitude) and use patsy to pull those columns out of the data.

    If `lat_long` is a tuple of arrays/vectors/DataFrames, I assume they're in the order
    (latitude, longitude) and use np.hstack to paste them together.

    Otherwise, just return things.
    """

    def _pull_string_cols(lat_long, data):
        lat_colname = lat_long[0]
        long_colname = lat_long[1]
        lat_long_patsy_form = '0 + ' + lat_colname + ' + ' + long_colname
        try:
            return dmatrix(lat_long_patsy_form, data)
        except KeyError:
            error_str = ("Tried to find latitude and longitude columns '{}' and '{}' in "
                         "data, but failed. Please provide a valid column name or simply"
                         " pass an array.".format(lat_colname, long_colname))
            raise KeyError(error_str)

    if isinstance(lat_long, (str, bytes, bytearray)):
        # If lat_long is a string, lat_long[0] and lat_long[1] will be string slices
        # (letters), not the variable names we're aiming for.
        raise ValueError("Please provide *two* strings for lat_long.")

    if isinstance(lat_long, (tuple, list)):
        if len(lat_long) != 2:
            raise ValueError("Expecting lat_long to be a tuple of 2 variables.")
        if (isinstance(lat_long[0], str) and isinstance(lat_long[1], str)):
            lat_long_array = _pull_string_cols(lat_long, data)
        elif (isinstance(lat_long[0], bytes) and isinstance(lat_long[1], bytes)):
            lat_long = [name.decode('utf-8') for name in lat_long]
            lat_long_array = _pull_string_cols(lat_long, data)
        else:
            # Else, we're probably given a tuple of two columns.  Check that they
            # seem okay and combine them.
            lat = lat_long[0]
            lon = lat_long[1]
            try:
                lat_shape = lat.shape
                lon_shape = lon.shape
            except AttributeError:
                err_msg = ("It looks like you provided latitude and longitude in a list "
                           "or tuple. In that case, I'm expecting the elements of the "
                           "tuple to be arrays, but these aren't.")
                raise ValueError(err_msg)

            # check that shapes conform:
            if len(lat_shape) < 2:
                # reshape so np.hstack works like I want
                lat = lat.reshape(-1, 1)
                lat_shape = lat.shape
            if len(lon_shape) < 2:
                lon = lon.reshape(-1, 1)
                lon_shape = lon.shape
            # and finally, combine to one array
            lat_long_array = np.hstack((lat, lon))
    else:  # i.e. not a list/tuple
        # don't bother with error checking here; that happens in check_geography anyway
        lat_long_array = lat_long
    return lat_long_array


def panel(formula_like, data, lat_long, time, group, dist_cutoff, time_cutoff = None,
          dist_kernel = 'uniform', time_kernel = 'bartlett'):
    """Calculate Conley standard errors for panel data.

    Parameters
    ----------
    formula_like : string or other Patsy formula
        e.g. "my_y_variable = my_X_var1 + my_X_var2"
        See http://patsy.readthedocs.io/en/latest/formulas.html#formulas for
        details on Patsy formulas.
    data : array-like
        Must contain all the variables referenced in the formula.
    lat_long : array_like, or tuple of names of columns in data
        An N-by-2 array of latitudes (in the first column) and longitudes (in
        the second column). Both latitude and longitude should be measured
        in degrees. Valid longitudes are [-90, 90].  Valid latitudes are
        (-180, 180]. The number of rows should be the same as the rows in data.
    time : array-like, or name of a column in data
        An N-row array of time observations, measured in whatever units
    group : array-like, or name of a column in data
        An N-row array of group identifiers
    dist_cutoff : number
        The maximum distance over which covariance is possible.
        cutoff must be a positive number in the range (0, 20015).
    time_cutoff : number  (default: T^(1/4))
        The maximum length of time over which covariance is possible.
    dist_kernel : string  (default: 'uniform')
        The kernel function to weight the distances.
        Valid options are:
        'bartlett', 'triangle', 'epanechnikov', 'quartic', 'biweight' and
        'triweight'. (Bartlett is the same as triangle. Quartic is the same as
        biweight.)
    time_kernel : string  (default: 'bartlett')
        The kernel function to weight the times.
        Valid options are:
        'bartlett', 'triangle', 'epanechnikov', 'quartic', 'biweight' and
        'triweight'. (Bartlett is the same as triangle. Quartic is the same as
        biweight.)
    """
    from cachey import Cache  # TODO: do I need caching?  (I don't think so?)
    cache = Cache(1e9)  # max cache is 1 GB  TODO: is this reasonable?
    y, X = dmatrices(formula_like, data, eval_env = 1, NA_action = 'raise')
    # TODO: handle cases where people provide weird formulas?

    if isinstance(time, str):
        try:
            time = data[time]
        except KeyError:
            error_str = ("Tried to find time-index column '{}' in data, but failed. "
                         "Please provide a valid column name or simply pass an "
                         "array.".format(time))
            raise KeyError(error_str)
    lat_long = parse_lat_long(lat_long, data)

    # Raise an exception if the data look funky
    nobs = check_parameters_panel(y, X, time, lat_long, dist_cutoff)

    # TODO: I have no idea if this leaf_size is reasonable.
    # If running out of memory, divide N by a larger number.
    # TODO: consider limiting available memory
    # 40 is the default.
    leaf_size = max(40, nobs // 1000)

    # TODO: use statsmodels OLS?
    betahat, _, rank, _ = np.linalg.lstsq(X, y)
    if rank != X.shape[1]:
        raise np.linalg.LinAlgError('X matrix is not full rank!')
    residuals = (y - X @ betahat)

    BallTree_cached = cache.memoize(BallTree)
    # TODO: allow for vincenty
    balltree = BallTree_cached(lat_long, metric = 'greatcircle', leaf_size = leaf_size)
    query_radius_cached = cache.memoize(balltree.query_radius)

    # START HERE.
    # The code in multiply_XeeX wasn't written for panel data.
    # Think about how I want to do the joint time/space deal.
    if dist_kernel == 'uniform':
        neighbors = query_radius_cached(lat_long, r = dist_cutoff)
        raise NotImplementedError()
        # filling = multiply_XeeX(neighbors, residuals, X, dist_kernel)
    else:
        neighbors, neighbor_distances = query_radius_cached(
            lat_long, r = dist_cutoff, return_distance = True)
        raise NotImplementedError()
        # filling = multiply_XeeX(neighbors, residuals, X, dist_kernel,
        #                         distances = neighbor_distances, cutoff = dist_cutoff)
        del neighbor_distances
    del balltree, neighbors, y, residuals

    bread = np.linalg.inv(X.T @ X)
    sandwich = nobs * (bread.T @ filling @ bread)
    se = np.sqrt(np.diag(sandwich)).reshape(-1, 1)
    return se


def neighbors_to_sparse_uniform(neighbors):
    nrow = len(neighbors)
    nnz = 0  # number of non-zeros
    for i_neighbor in range(nrow):
        nnz += len(neighbors[i_neighbor])
    rows = np.empty(nnz, dtype=ITYPE)
    cols = np.empty(nnz, dtype=ITYPE)
    vals = np.ones(nnz,  dtype=DTYPE)  # noqa: E241
    sparse_idx = 0
    for row_idx in range(nrow):
        neighbors_row = neighbors[row_idx]
        n_neighbors = len(neighbors_row)
        end_idx = sparse_idx + n_neighbors
        rows[sparse_idx: end_idx] = row_idx
        cols[sparse_idx: end_idx] = neighbors_row
        sparse_idx += n_neighbors
    assert rows.shape[0] == cols.shape[0] == nnz
    # Finally, we construct a regular SciPy sparse matrix:
    return coo_matrix((vals, (rows, cols)), shape=(nrow, nrow)).tocsr()


def neighbors_to_sparse_nonuniform(neighbors, kernel, distances, cutoff):
    kernel_fn = get_kernel_fn(kernel)
    nrow = len(neighbors)

    nnz = 0  # number of non-zeros
    for i_neighbor in range(nrow):
        nnz += len(neighbors[i_neighbor])
    rows = np.empty(nnz, dtype=ITYPE)
    cols = np.empty(nnz, dtype=ITYPE)
    vals = np.empty(nnz, dtype=DTYPE)
    sparse_idx = 0
    for row_idx in range(nrow):
        neighbors_row = neighbors[row_idx]
        n_neighbors = len(neighbors_row)
        end_idx = sparse_idx + n_neighbors
        rows[sparse_idx: end_idx] = row_idx
        cols[sparse_idx: end_idx] = neighbors_row
        vals[sparse_idx: end_idx] = kernel_fn(distances[row_idx], cutoff)
        sparse_idx += n_neighbors
    assert rows.shape[0] == cols.shape[0] == vals.shape[0] == nnz

    # Finally, we construct a regular SciPy sparse matrix:
    return coo_matrix((vals, (rows, cols)), shape=(nrow, nrow)).tocsr()


def neighbors_to_sparse(neighbors, kernel = 'uniform', distances = None, cutoff = None):
    if kernel == 'uniform':
        if cutoff is not None or distances is not None:
            err_msg = ("this combination of parameters should never be "
                       "necessary; it's a coding mistake")
            raise ValueError(err_msg)
        neighbors_sparse = neighbors_to_sparse_uniform(neighbors)
    else:
        if cutoff is None or distances is None:
            err_msg = ("this combination of parameters should never be "
                       "necessary; it's a coding mistake")
            raise ValueError(err_msg)
        if len(neighbors) != len(distances):
            err_msg = "Number of neighbors and distances don't match."
            raise ValueError(err_msg)
        neighbors_sparse = neighbors_to_sparse_nonuniform(
            neighbors, kernel, distances, cutoff)
    return neighbors_sparse
