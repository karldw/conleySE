
# coding: utf-8
"""Tests, copied directly from scikit-learn.

Removed __init__ from TestMetrics.
Removed pickling tests.
"""

import itertools
import pickle

import numpy as np
from numpy.testing import assert_array_almost_equal

import scipy
from scipy.spatial.distance import cdist
from sklearn.neighbors.dist_metrics import DistanceMetric
from nose import SkipTest


def dist_func(x1, x2, p):
    return np.sum((x1 - x2) ** p) ** (1. / p)


def cmp_version(version1, version2):
    version1 = tuple(map(int, version1.split('.')[:2]))
    version2 = tuple(map(int, version2.split('.')[:2]))

    if version1 < version2:
        return -1
    elif version1 > version2:
        return 1
    else:
        return 0


class TestMetrics:
    # def __init__(self, n1=20, n2=25, d=4, zero_frac=0.5,
    #              rseed=0, dtype=np.float64):
    n1 = 20
    n2 = 25
    d = 4
    zero_frac = 0.5
    rseed = 0
    dtype = np.float64
    np.random.seed(rseed)
    X1 = np.random.random((n1, d)).astype(dtype)
    X2 = np.random.random((n2, d)).astype(dtype)

    # make boolean arrays: ones and zeros
    X1_bool = X1.round(0)
    X2_bool = X2.round(0)

    V = np.random.random((d, d))
    VI = np.dot(V, V.T)

    metrics = {'euclidean': {},
               'cityblock': {},
               'minkowski': dict(p=(1, 1.5, 2, 3)),
               'chebyshev': {},
               'seuclidean': dict(V=(np.random.random(d),)),
               'wminkowski': dict(p=(1, 1.5, 3),
                                  w=(np.random.random(d),)),
               'mahalanobis': dict(VI=(VI,)),
               'hamming': {},
               'canberra': {},
               'braycurtis': {}}

    bool_metrics = ['matching', 'jaccard', 'dice',
                    'kulsinski', 'rogerstanimoto', 'russellrao',
                    'sokalmichener', 'sokalsneath']

    def test_cdist(self):
        for metric, argdict in self.metrics.items():
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                D_true = cdist(self.X1, self.X2, metric, **kwargs)
                yield self.check_cdist, metric, kwargs, D_true

        for metric in self.bool_metrics:
            D_true = cdist(self.X1_bool, self.X2_bool, metric)
            yield self.check_cdist_bool, metric, D_true

    def check_cdist(self, metric, kwargs, D_true):
        if metric == 'canberra' and cmp_version(scipy.__version__, '0.9') <= 0:
            raise SkipTest("Canberra distance incorrect in scipy < 0.9")
        dm = DistanceMetric.get_metric(metric, **kwargs)
        D12 = dm.pairwise(self.X1, self.X2)
        assert_array_almost_equal(D12, D_true)

    def check_cdist_bool(self, metric, D_true):
        dm = DistanceMetric.get_metric(metric)
        D12 = dm.pairwise(self.X1_bool, self.X2_bool)
        assert_array_almost_equal(D12, D_true)

    def test_pdist(self):
        for metric, argdict in self.metrics.items():
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                D_true = cdist(self.X1, self.X1, metric, **kwargs)
                yield self.check_pdist, metric, kwargs, D_true

        for metric in self.bool_metrics:
            D_true = cdist(self.X1_bool, self.X1_bool, metric)
            yield self.check_pdist_bool, metric, D_true

    def check_pdist(self, metric, kwargs, D_true):
        if metric == 'canberra' and cmp_version(scipy.__version__, '0.9') <= 0:
            raise SkipTest("Canberra distance incorrect in scipy < 0.9")
        dm = DistanceMetric.get_metric(metric, **kwargs)
        D12 = dm.pairwise(self.X1)
        assert_array_almost_equal(D12, D_true)

    def check_pdist_bool(self, metric, D_true):
        dm = DistanceMetric.get_metric(metric)
        D12 = dm.pairwise(self.X1_bool)
        assert_array_almost_equal(D12, D_true)

    # def test_pickle(self):
    #     for metric, argdict in self.metrics.items():
    #         keys = argdict.keys()
    #         for vals in itertools.product(*argdict.values()):
    #             kwargs = dict(zip(keys, vals))
    #             yield self.check_pickle, metric, kwargs
    #
    #     for metric in self.bool_metrics:
    #         yield self.check_pickle_bool, metric

    def check_pickle_bool(self, metric):
        dm = DistanceMetric.get_metric(metric)
        D1 = dm.pairwise(self.X1_bool)
        dm2 = pickle.loads(pickle.dumps(dm))
        D2 = dm2.pairwise(self.X1_bool)
        assert_array_almost_equal(D1, D2)

    # def check_pickle(self, metric, kwargs):
    #     dm = DistanceMetric.get_metric(metric, **kwargs)
    #     D1 = dm.pairwise(self.X1)
    #     dm2 = pickle.loads(pickle.dumps(dm))
    #     D2 = dm2.pairwise(self.X1)
    #     assert_array_almost_equal(D1, D2)


def test_haversine_metric():
    def haversine_slow(x1, x2):
        return 2 * np.arcsin(np.sqrt(np.sin(0.5 * (x1[0] - x2[0])) ** 2 +
                                     np.cos(x1[0]) * np.cos(x2[0]) *
                                     np.sin(0.5 * (x1[1] - x2[1])) ** 2))

    X = np.random.random((10, 2))

    haversine = DistanceMetric.get_metric("haversine")

    D1 = haversine.pairwise(X)
    D2 = np.zeros_like(D1)
    for i, x1 in enumerate(X):
        for j, x2 in enumerate(X):
            D2[i, j] = haversine_slow(x1, x2)

    assert_array_almost_equal(D1, D2)
    assert_array_almost_equal(haversine.dist_to_rdist(D1),
                              np.sin(0.5 * D2) ** 2)


def test_pyfunc_metric():
    X = np.random.random((10, 3))

    euclidean = DistanceMetric.get_metric("euclidean")
    pyfunc = DistanceMetric.get_metric("pyfunc", func=dist_func, p=2)

    # Check if both callable metric and predefined metric initialized
    # DistanceMetric object is picklable
    euclidean_pkl = pickle.loads(pickle.dumps(euclidean))
    pyfunc_pkl = pickle.loads(pickle.dumps(pyfunc))

    D1 = euclidean.pairwise(X)
    D2 = pyfunc.pairwise(X)

    D1_pkl = euclidean_pkl.pairwise(X)
    D2_pkl = pyfunc_pkl.pairwise(X)

    assert_array_almost_equal(D1, D2)
    assert_array_almost_equal(D1_pkl, D2_pkl)
