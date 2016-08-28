# import itertools
# import pickle

# Cludge to import the correct module in the build path.
import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from hypothesis import given, assume  # nopep8
from hypothesis.strategies import floats, tuples  # nopep8

import numpy as np  # nopep8
from numpy.testing import assert_array_almost_equal  # nopep8
from geopy.distance import great_circle as geopy_great_circle  # nopep8
from geopy.distance import vincenty as geopy_vincenty  # nopep8
# from hypothesis.extra import numpy
# import scipy
# from scipy.spatial.distance import cdist
from dist_metrics import DistanceMetric  # nopep8
# EARTH_RADIUS is the same value used in dist_metrics
from geopy.distance import EARTH_RADIUS  # nopep8
# from sklearn.neighbors.dist_metrics import DistanceMetric
# from nose import SkipTest

hypoth_lat_long = tuples(floats(min_value = -90, max_value = 90),
                         floats(min_value = -179.999999999999999, max_value = 180))


@given(hypoth_lat_long, hypoth_lat_long)
def test_great_circle_metric(pt1, pt2):
    X = np.array([pt1, pt2])

    great_circle = DistanceMetric.get_metric("greatcircle")

    D1 = great_circle.pairwise(X)
    D2 = np.zeros_like(D1)
    for i, x1 in enumerate(X):
        for j, x2 in enumerate(X):
            D2[i, j] = geopy_great_circle(x1, x2).km

    if EARTH_RADIUS == 6371.009:
        # geopy is changing its EARTH_RADIUS and I've set up with the newer one
        assert_array_almost_equal(D1, D2)
    # but in either case, can run this version:
    assert_array_almost_equal(great_circle.dist_to_rdist(D1),
                              D2 / EARTH_RADIUS)


@given(hypoth_lat_long, hypoth_lat_long)
def test_vincenty_metric(pt1, pt2):
    X = np.array([pt1, pt2])

    vincenty = DistanceMetric.get_metric("vincenty")
    try:
        D1 = vincenty.pairwise(X)
    except ValueError:  # when vincenty fails to converge, don't worry about it
        assume(0)
    D2 = np.zeros_like(D1)
    for i, x1 in enumerate(X):
        for j, x2 in enumerate(X):
            try:
                D2[i, j] = geopy_vincenty(x1, x2).km
            except UnboundLocalError:
                D2[i, j] = 0.0
    # We end up with a 2x2 matrix of point comparisons.  If that's not symmetric,
    # something has gone wrong (and hypothesis finds these things).
    if (D2.transpose() != D2).any():
        assume(0)
    assert_array_almost_equal(D1, D2)
