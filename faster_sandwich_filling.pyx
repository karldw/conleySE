# -*- coding: utf-8 -*-
# cython: profile=True
# import cython
cimport cython
import numpy as np
cimport numpy as np
from typedefs cimport DTYPE_t, ITYPE_t

from typedefs import DTYPE, ITYPE
from libc.math cimport fabs, cos, M_PI_2, M_PI_4, pow
np.import_array()



class GeographyError(ValueError):
    pass
class CutoffError(GeographyError):
    pass


def get_kernel_fn(kernel):
    """Get a kernel for the pointwise distances."""
    # Note, these cython kernels are only ~ 3-4x faster than the numpy version.
    # It might not be worth the hassle.
    kernel_dict = {'bartlett': bartlett,
                   'Bartlett': bartlett,
                   'triangle': bartlett,
                   'Epanechnikov': epanechnikov,
                   'epanechnikov': epanechnikov,
                   'quartic': biweight,
                   'biweight': biweight,
                   'triweight': triweight,
                   'tricube': tricube,
                   'cosine': cosine,
    }

    if callable(kernel):
        return kernel
    try:
        return kernel_dict[kernel]
    except KeyError:
        known_kernels = set([x for x in kernel_dict.keys()])
        error_message = "Unknown kernel specified. Please provide one of these, or your own function: {}".format(known_kernels)
        raise KeyError(error_message)


@cython.embedsignature(True)  # embed function signature in docstring
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative indexing
@cython.initializedcheck(False)  # don't check that data are initialized
@cython.cdivision(True)  # revert to the faster C division rules (make sure your code never divides by zero or a negative number!)
cpdef DTYPE_t[:] bartlett(DTYPE_t[:] dists, DTYPE_t cutoff):
    """Weight distances by the Bartlett (triangular) kernel.

    Important: The function _does not_ check weather the distance is outside the
    cutoff.  You should do this elsewhere.

    Input: A memoryview of distances (float64) and a cutoff (float64).
    Output: A memoryview of weights (float64) in the range [0, 1].
    """
    cdef ITYPE_t i
    cdef DTYPE_t u
    cdef DTYPE_t[:] weights = np.empty_like(dists)
    with nogil:
        for i in range(dists.shape[0]):
            u = dists[i] / cutoff
            weights[i] = 1 - fabs(u)
    return weights


@cython.embedsignature(True)  # embed function signature in docstring
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative indexing
@cython.initializedcheck(False)  # don't check that data are initialized
@cython.cdivision(True)  # revert to the faster C division rules (make sure your code never divides by zero or a negative number!)
cpdef DTYPE_t[:] epanechnikov(DTYPE_t[:] dists, DTYPE_t cutoff):
    """Weight distances by the Epanechnikov kernel.

    Important: The function _does not_ check weather the distance is outside the
    cutoff.  You should do this elsewhere.

    Input: A memoryview of distances (float64) and a cutoff (float64).
    Output: A memoryview of weights (float64) in the range [0, 1].
    """
    cdef ITYPE_t i
    cdef DTYPE_t u
    # cdef DTYPE_t inv_cutoff_sq, multiplier
    cdef DTYPE_t[:] weights = np.empty_like(dists)
    with nogil:
        # inv_cutoff_sq = 1 / (cutoff**2)
        # multiplier = 0.75 / cutoff
        for i in range(dists.shape[0]):
            u = dists[i] / cutoff
            weights[i] = 0.75 * (1 - pow(u, 2))
    return weights


# Commented out for debugging.
# @cython.embedsignature(True)  # embed function signature in docstring
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative indexing
# @cython.initializedcheck(False)  # don't check that data are initialized
# @cython.cdivision(True)
cpdef DTYPE_t[:] biweight(DTYPE_t[:] dists, DTYPE_t cutoff):
    """Weight distances by the biweight (quartic) kernel.

    Important: The function _does not_ check weather the distance is outside the
    cutoff.  You should do this elsewhere.

    Input: A memoryview of distances (float64) and a cutoff (float64).
    Output: A memoryview of weights (float64) in the range [0, 1].
    """
    raise NotImplementedError
    # The results here are always zero -- why?
    cdef ITYPE_t i
    cdef DTYPE_t u
    cdef DTYPE_t[:] weights = np.empty_like(dists)
    with nogil:
        for i in range(dists.shape[0]):
            u = dists[i] / cutoff
            weights[i] = 15 / 16 * pow(1 - pow(u, 2), 2)
    return weights


# Commented out for debugging.
# @cython.embedsignature(True)  # embed function signature in docstring
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative indexing
# @cython.initializedcheck(False)  # don't check that data are initialized
# revert to the faster C division rules
# (make sure your code never divides by zero or a negative number!)
# @cython.cdivision(True)
cpdef DTYPE_t[:] triweight(DTYPE_t[:] dists, DTYPE_t cutoff):
    """Weight distances by the triweight kernel.

    Important: The function _does not_ check weather the distance is outside the
    cutoff.  You should do this elsewhere.

    Input: A memoryview of distances (float64) and a cutoff (float64).
    Output: A memoryview of weights (float64) in the range [0, 1].
    """
    raise NotImplementedError
    # The results here are off by a factor of 32/35, as if they're not
    # being properly multiplied -- why?
    cdef ITYPE_t i
    cdef DTYPE_t u
    cdef DTYPE_t[:] weights = np.empty_like(dists)
    with nogil:
        for i in range(dists.shape[0]):
            u = dists[i] / cutoff
            weights[i] = (35 / 32) * pow(1 - pow(u, 2), 3)
    return weights


# Commented out for debugging.
# @cython.embedsignature(True)  # embed function signature in docstring
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative indexing
# @cython.initializedcheck(False)  # don't check that data are initialized
# revert to the faster C division rules
# (make sure your code never divides by zero or a negative number!)
# @cython.cdivision(True)
cpdef DTYPE_t[:] tricube(DTYPE_t[:] dists, DTYPE_t cutoff):
    """Weight distances by the tricube kernel.

    Important: The function _does not_ check weather the distance is outside the
    cutoff.  You should do this elsewhere.

    Input: A memoryview of distances (float64) and a cutoff (float64).
    Output: A memoryview of weights (float64) in the range [0, 1].
    """
    raise NotImplementedError
    # The results here are always zero -- why?
    cdef ITYPE_t i
    cdef DTYPE_t u
    cdef DTYPE_t[:] weights = np.empty_like(dists)
    with nogil:
        for i in range(dists.shape[0]):
            u = dists[i] / cutoff
            weights[i] = (70 / 81) * pow(1 - pow(fabs(u), 3), 3)
    return weights


# Commented out for debugging.
@cython.embedsignature(True)  # embed function signature in docstring
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative indexing
@cython.initializedcheck(False)  # don't check that data are initialized
# revert to the faster C division rules
# (make sure your code never divides by zero or a negative number!)
@cython.cdivision(True)
cpdef DTYPE_t[:] cosine(DTYPE_t[:] dists, DTYPE_t cutoff):
    """Weight distances by the cosine kernel.

    Important: The function _does not_ check weather the distance is outside the
    cutoff.  You should do this elsewhere.

    Input: A memoryview of distances (float64) and a cutoff (float64).
    Output: A memoryview of weights (float64) in the range [0, 1].
    """
    cdef ITYPE_t i
    cdef DTYPE_t u
    cdef DTYPE_t[:] weights = np.empty_like(dists)
    with nogil:
        for i in range(dists.shape[0]):
            u = dists[i] / cutoff
            weights[i] = M_PI_4 * cos(M_PI_2 * u)
    return weights


def multiply_XeeX(
        object neighbors,
        np.ndarray[DTYPE_t, ndim = 1, mode = 'c'] residuals,
        np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] X,
        str kernel,
        object distances = None,
        cutoff = None):

    # neighbors is a numpy ndarray (dtype = 'O') of ndarrays
    cdef DTYPE_t[:, :] result
    if kernel == 'uniform':
        result = multiply_XeeX_uniform(neighbors, residuals, X)
    else:
        result = multiply_XeeX_NONuniform(neighbors, residuals, X, kernel, distances, cutoff)
    return result

# Commented out for debugging.
# @cython.embedsignature(True)  # embed function signature in docstring
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative indexing
# @cython.initializedcheck(False)  # don't check that data are initialized
# revert to the faster C division rules
# (make sure your code never divides by zero or a negative number!)
# @cython.cdivision(True)
cdef inline DTYPE_t[:, :] multiply_XeeX_uniform(
        object neighbors,  # an numpy ndarray (dtype = 'O') of ndarrays
        DTYPE_t[:] residuals,
        DTYPE_t[:, :] X):
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t K = X.shape[1]
    # i indexes rows, j indexes neighbors as does neighbor_j
    # p and q index the extra for-loops I need to fill memoryviews
    cdef ITYPE_t i, j, p, q, neighbor_j
    cdef DTYPE_t e_i_div_N, e_j
    cdef DTYPE_t[:, :] X_iT_ei
    cdef DTYPE_t[:] X_i, X_neighborj
    cdef DTYPE_t[:, :] tempsum_X_j_ej
    cdef ITYPE_t[:] neighbors_i
    cdef DTYPE_t[:, :] output = np.zeros((K, K), dtype = DTYPE)
    cdef DTYPE_t[:, :] zeros_row = np.zeros((1, K), dtype = DTYPE)
    cdef DTYPE_t[:, :] zeros_col_K = zeros_row.T
    # Calculate sum_i^N [ X_i * e_i * (sum_j^n X_j' * e_j) ]
    # for i and j near one another
    # 'bint' is a cython type, short for 'boolean int' to make C and python
    # booleans play nicely.
    cdef bint every_point_is_a_neighbor_of_every_other = True
    # TODO: is it better to convert neighbors to a sparse matrix and use scipy's
    # sparse multiplication?
    for i in range(N):
        # neighbors is a weird N-length ndarray of variable-length ndarrays,
        # so have to use the GIL to parse.
        neighbors_i = neighbors[i]
        if neighbors_i.shape[0] < N:
            every_point_is_a_neighbor_of_every_other = False
        X_iT_ei = zeros_col_K.copy()  # awful variable name stands for X[i].T * e[i]
        # TODO: use copy_fortran If I decinde I want the fortran-strided version.
        tempsum_X_j_ej = zeros_row.copy()
        with nogil:
            e_i_div_N = residuals[i] / N
            X_i = X[i]
            for p in range(K):
                # assign to the rows, so it's effectively X[i, np.newaxis].T * e[i] / N
                X_iT_ei[p, 0] = X_i[p] * e_i_div_N
            # Same as the previous two lines, using numpy's array broadcasting:
            # X_i_ei = X[i, np.newaxis].T * e_i
            for j in range(neighbors_i.shape[0]):
                neighbor_j = neighbors_i[j]
                X_neighborj = X[neighbor_j]
                e_j = residuals[neighbor_j]

                # memoryviews don't have numpy's broadcasting, so have to loop here.
                # is this faster? maybe...
                for q in range(K):
                    tempsum_X_j_ej[0, q] += X_neighborj[q] * e_j
                # Same as the previous two lines, using numpy's array broadcasting:
                # tempsum_X_j_ej += X[neighbor_j, None] * e_j  # Using None is equivalent to np.newaxis, but allows compiling.
        # Need the GIL back for the matrix multiplication:
        output += np.dot(X_iT_ei, tempsum_X_j_ej)
    if every_point_is_a_neighbor_of_every_other:
        raise CutoffError("Every point is a neighbor of every other. You must use a smaller cutoff value.")
    return output #/ N  # would normally divide by N here, but it's easier to just divide e_i by N above.


# Commented out for debugging.
# @cython.embedsignature(True)  # embed function signature in docstring
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative indexing
# @cython.initializedcheck(False)  # don't check that data are initialized
# @cython.cdivision(True)  # revert to the faster C division rules (make sure your code never divides by zero or a negative number!)
cdef inline DTYPE_t[:, :] multiply_XeeX_NONuniform(
        object neighbors,
        DTYPE_t[:] residuals,
        DTYPE_t[:, :] X,
        str kernel,
        object distances,
        DTYPE_t cutoff):
    assert cutoff is not None
    kernel_fn = get_kernel_fn(kernel)
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t K = X.shape[1]
    # i indexes rows, j indexes neighbors as does neighbor_j,
    # p and q index the random for-loops I need to fill memoryviews
    cdef ITYPE_t i, j, p, q, neighbor_j
    cdef DTYPE_t e_i_div_N, e_j
    cdef DTYPE_t[:, :] X_iT_ei
    cdef DTYPE_t[:] X_i, X_neighborj
    cdef DTYPE_t[:, :] tempsum_X_j_ej_weight
    cdef ITYPE_t[:] neighbors_i
    cdef DTYPE_t[:] distances_i, weights_i
    cdef DTYPE_t[:, :] zeros_row = np.zeros((1, K), dtype = DTYPE)
    cdef DTYPE_t[:, :] output = np.zeros((K, K), dtype = DTYPE)
    cdef DTYPE_t[:, :] zeros_col_K = zeros_row.T
    # Calculate sum_i^N [ X_i * e_i * (sum_j^n X_j' * e_j * weights_i,j) ]
    # for i and j near one another
    cdef bint every_point_is_a_neighbor_of_every_other = True
    for i in range(N):
        # neighbors is a weird N-length ndarray of variable-length ndarrays, so have to use the GIL to parse.
        neighbors_i = neighbors[i]
        if neighbors_i.shape[0] < N:
            every_point_is_a_neighbor_of_every_other = False
        distances_i = distances[i]
        weights_i = kernel_fn(distances_i, cutoff)
        X_iT_ei = zeros_col_K.copy()  # awful variable name stands for X[i].T * e[i]
        tempsum_X_j_ej_weight = zeros_row.copy()  # TODO(?) copy_fortran for the fortran-strided version
        with nogil:
            e_i_div_N = residuals[i] / N
            X_i = X[i]
            for p in range(K):
                X_iT_ei[p, 0] = X[i][p] * e_i_div_N  # assign to the rows, so it's effectively X[i, np.newaxis].T * e[i] / N
            # Same as the previous two lines, using numpy's array broadcasting
            # X_i_ei = X[i, None].T * e_i  # Using None is equivalent to np.newaxis, but allows compiling.
            for j in range(neighbors_i.shape[0]):
                neighbor_j = neighbors_i[j]
                X_neighborj = X[neighbor_j]
                e_j = residuals[neighbor_j]
                for q in range(K):  # element-by-element:
                    tempsum_X_j_ej_weight[0, q] += X_neighborj[q] * e_j * weights_i[j]
                # Same as the previous two lines, using numpy's array broadcasting:
                # tempsum_X_j_ej_weight += X[neighbor_j, None] * e_j * weights_i  # (Using None is equivalent to np.newaxis, but allows compiling.)
        # Need the GIL back for the matrix multiplication:
        output += np.dot(X_iT_ei, tempsum_X_j_ej_weight)
    if every_point_is_a_neighbor_of_every_other:
        raise CutoffError("Every point is a neighbor of every other. You must use a smaller cutoff value.")
    return output #/ N  # would normally divide by N here, but it's easier to just divide e_i by N above.
